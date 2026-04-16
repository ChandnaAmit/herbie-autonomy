import cv2
import os
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
import vlc
import serial, time

PORT = "/dev/ttyACM0"
BAUD = 115200

TRACKER_YAML = "/home/amit/bytetrack.yaml"
YOLO_MODEL   = "yolo12m.engine"  # TensorRT engine
YUNET_PATH = "/home/amit/yunet.onnx"
FACE_CONF_THRESHOLD = 0.6

# 848x480 keeps color==depth after align()
COLOR_W, COLOR_H = 848, 480
DEPTH_W, DEPTH_H = 848, 480

# --- Settings (keep/adjust as needed) ---
TARGET_DIST_M   = 1.0
STEER_K         = 0.9
THROTTLE_K      = 0.5
THROTTLE_STEER_K = 0.8
THROTTLE_STEP_UP = 0.05
THROTTLE_STEP_DOWN = 0.25
STEER_CENTER_US = 1400
STEER_SPAN_US   = 300
STEER_MIN_US    = 1100
STEER_MAX_US    = 1700
THROTTLE_MIN = 0.0  # per  	firmware: anything below 0.08 is ignored

# throttle command rate limit (to avoid flooding Arduino)
CMD_RATE_HZ     = 30.0

def open_port():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(1.0)
    ser.reset_input_buffer()
    return ser

def send_line(ser, line, timeout=1.5):
    ser.write((line.strip() + "\n").encode("ascii"))
    ser.flush()
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        resp = ser.readline().decode(errors="ignore").strip()
        if not resp: 	
            continue
        last = resp
        if resp.startswith(("OK", "ERR")):
            return resp
    return last or "<no response>"


def play_sound(filename):
    try:
        p = vlc.MediaPlayer(filename)
        p.play()
        return p  # return handle so we can stop later
    except Exception as e:
        print(f"[WARN] VLC playback failed: {e}")
        return None

        
def check_top_dark_ratio(bgr_frame, send_cmd,
                         top_frac=0.2, bottom_frac=0.2,
                         on_ratio=0.55,    # turn HL20 on if top is <60% of bottom
                         off_ratio=0.68,   # go back to HL5 if top is >85% of bottom
                         cooldown_s=0.5,   # min time between changes
                         on_level=20, restore_level=5, vis=None):
    H, W = bgr_frame.shape[:2]
    ht = max(1, int(H * top_frac))
    hb = max(1, int(H * bottom_frac))

    top_roi = bgr_frame[0:ht, :]
    bot_roi = bgr_frame[H - hb:H, :]

    # use HSV V means 
    v_top = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    v_bot = cv2.cvtColor(bot_roi, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    ratio = float(v_top / (v_bot + 1e-6))  # avoid div-by-zero

    # tiny persistent state
    now = time.time()
    st = getattr(check_top_dark_ratio, "_state", None)
    if st is None:
        st = {"hl_on": False, "last_t": 0.0}
        check_top_dark_ratio._state = st

    if (now - st["last_t"]) >= cooldown_s:
        if not st["hl_on"] and ratio < on_ratio:
            send_cmd(f"HL {on_level}")
            st["hl_on"] = True
            st["last_t"] = now
        elif st["hl_on"] and ratio > off_ratio:
            send_cmd(f"HL {restore_level}")
            st["hl_on"] = False
            st["last_t"] = now

    if vis is not None:
        light_status = "Lights:ON" if st["hl_on"] else "Lights:DRL"
        put_label_with_bg(vis, f"{light_status} {ratio:.2f} [{on_ratio:.2f}/{off_ratio:.2f}]", 600, 14, scale=0.5, thickness=1)

    return ratio, st["hl_on"]


def put_label_with_bg(img, text, x, y, scale=0.6, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    pad = 0
    x1 = max(0, x - pad); y1 = max(0, y - th - pad)
    x2 = min(img.shape[1] - 1, x + tw + pad); y2 = min(img.shape[0] - 1, y + pad)
    cv2.putText(img, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

def nearest_valid(depth_m, cx, cy, max_radius=40, step=4):
    H, W = depth_m.shape
    for r in range(step, max_radius + 1, step):
        x1 = max(0, cx - r); x2 = min(W, cx + r)
        y1 = max(0, cy - r); y2 = min(H, cy + r)
        patch = depth_m[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        v = np.nanmedian(patch)
        if np.isfinite(v):
            return float(v)
    return float("nan")

def get_center_patch_rect(bbox_xyxy, img_shape_hw, patch_frac=0.25, min_size=6):
    """Return (xs, ys, xe, ye) for the center patch used for depth measurement."""
    H, W = img_shape_hw
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1 = np.clip(x1, 0, W - 1); x2 = np.clip(x2, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1); y2 = np.clip(y2, 0, H - 1)
    w = max(1, x2 - x1); h = max(1, y2 - y1)

    cx = x1 + w // 2; cy = y1 + h // 3
    pw = max(min_size, int(w * patch_frac))
    ph = max(min_size, int(h * patch_frac))
    xs = max(0, cx - pw // 2); xe = min(W, cx + pw // 2)
    ys = max(0, cy - ph // 2); ye = min(H, cy + ph // 2)
    if xe <= xs: xe = min(W, xs + min_size)
    if ye <= ys: ye = min(H, ys + min_size)
    return int(xs), int(ys), int(xe), int(ye)

def median_depth_in_bbox(depth_m, bbox_xyxy, img_shape_hw, patch_frac=0.25):
    # compute patch in *color/vis* coordinates (same as depth when not decimated)
    xs, ys, xe, ye = get_center_patch_rect(bbox_xyxy, img_shape_hw, patch_frac=patch_frac, min_size=6)
    patch = depth_m[ys:ye, xs:xe]
    if patch.size == 0:
        return float("nan"), (xs, ys, xe, ye)

    m = np.nanmedian(patch)
    if np.isfinite(m):
        return float(m), (xs, ys, xe, ye)

    # fallback around center
    cx = (xs + xe) // 2; cy = (ys + ye) // 2
    return nearest_valid(depth_m, cx, cy, max_radius=40, step=4), (xs, ys, xe, ye)

def detect_obstruction_in_poly(depth_m, pts, max_dist_m=1.0, min_pixels=20):
    # Ensure integer vertices
    poly = np.array(pts, dtype=np.int32)

    h, w = depth_m.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Fill polygon mask with 1s
    cv2.fillPoly(mask, [poly], 1)

    # Extract just the polygon region
    patch = depth_m.copy()
    patch[mask == 0] = np.nan

    # Valid finite depths
    valid = np.isfinite(patch)

    if not np.any(valid):
        # Nothing measurable in that region
        return False

    # Find pixels closer than threshold
    close = valid & (patch < max_dist_m)

    num_close = np.count_nonzero(close)
    if num_close >= min_pixels:
        # True obstruction: many pixels close
        min_dist = float(np.nanmin(patch))
        return True

    # Not enough close pixels → treat as no obstruction
    return False


def draw_blue_face_on_target(vis, face_detector, x1, y1, x2, y2):
    """
    Detect face inside the target person box and draw one blue face box on vis.
    Returns True if a face was drawn, else False.
    """
    H, W = vis.shape[:2]

    # Clamp ROI to image bounds
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W - 1, int(x2)))
    y2 = max(0, min(H - 1, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return False

    # Use upper part of person box for face search
    roi_y2 = int(y1 + 0.45 * (y2 - y1))
    roi_y2 = max(y1 + 1, min(H, roi_y2))

    person_roi = vis[y1:roi_y2, x1:x2]
    if person_roi.size == 0:
        return False

    rh, rw = person_roi.shape[:2]
    face_detector.setInputSize((rw, rh))

    faces = face_detector.detect(person_roi)

    if faces[1] is None or len(faces[1]) == 0:
        return False

    # Pick largest detected face in ROI
    best_face = max(faces[1], key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = best_face[:4].astype(int)

    # Convert ROI coords back to full-image coords
    gx1 = x1 + fx
    gy1 = y1 + fy
    gx2 = gx1 + fw
    gy2 = gy1 + fh

    # Draw blue rectangle on vis (OpenCV uses BGR)
    face_roi = vis[gy1:gy2, gx1:gx2]

    if face_roi.size > 0:
        # strong blur (odd kernel size required)
        k = 31  # increase for more blur (51, 71 if needed)
        blurred = cv2.GaussianBlur(face_roi, (k, k), 0)

        vis[gy1:gy2, gx1:gx2] = blurred

    # optional blue border
    #cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

    return True
    
    
# Draw arc path based on err
def draw_path_lines(vis, err, color=(80, 200, 80), thickness=2):
    h, w = vis.shape[:2]

    # where the path begins near the car
    y_bottom = h - 1

    # where the path fades into distance
    y_top = h - 160

    # width of the path near the car and far away
    half_width_bottom = 150
    half_width_top = 50

    # center of the vehicle/path near bottom
    cx0 = w // 2

    # steering curvature strength
    curve_gain = 200.0

    steps = 40
    left_pts = []
    right_pts = []

    for i in range(steps + 1):
        t = i / steps   # 0 near car, 1 far away

        # y goes upward into distance
        y = int(y_bottom - t * (y_bottom - y_top))

        # path gets narrower with distance -> perspective
        half_width = (1 - t) * half_width_bottom + t * half_width_top

        # curve grows with distance
        curve = curve_gain * err * (t ** 2)

        # optional small extra bend so it feels more natural
        curve += 20.0 * err * (t ** 3)

        center_x = cx0 + curve

        xl = int(center_x - half_width)
        xr = int(center_x + half_width)

        xl = max(0, min(w - 1, xl))
        xr = max(0, min(w - 1, xr))
        y = max(0, min(h - 1, y))

        left_pts.append([xl, y])
        right_pts.append([xr, y])

    left_pts = np.array(left_pts, dtype=np.int32).reshape((-1, 1, 2))
    right_pts = np.array(right_pts, dtype=np.int32).reshape((-1, 1, 2))

    cv2.polylines(vis, [left_pts], False, color, thickness)
    cv2.polylines(vis, [right_pts], False, color, thickness)
    
    
# Optional: custom action when a brand new ID appears in a frame
def on_new_detection(det):
    tid = det["id"]
    cls_id = det["class_id"]
    dist = det["distance_m"]
    print(f"[NEW] id={tid} class={cls_id} dist={dist}")
    #if cls_id == 0:
        #play_sound("/home/amit/Music/HORNS2.mp3")
    #pass
    
# Realsense Reset - Function not used and does not help  
def reset_realsense():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[ERR] No RealSense devices found for reset")
        return False
    dev = devices[0]
    print("[RS] Hardware reset triggered...")
    dev.hardware_reset()
    time.sleep(3)  # allow camera to re-enumerate
    return True

###################################
############### MAIN ##############
###################################
def main():
    # open Arduino serial once
    ser = open_port()

    # simple rate limiter for commands
    last_cmd_ts = 0.0
    min_dt = 1.0 / CMD_RATE_HZ

    max_g_force = 0.0
    left_pwr_state = 0.0
    right_pwr_state = 0.0
        
    def send_cmd(line: str):
        nonlocal last_cmd_ts
        now = time.time()
        if now - last_cmd_ts < min_dt:
            return
        resp = send_line(ser, line)
        #print(f">> {line}  << {resp}")
        last_cmd_ts = now
        
    def send_burst(lines, log=False, gap_s=0.01):
        resp = None
        for i, ln in enumerate(lines):
            resp = send_line(ser, ln)  # no rate limit for burst
            #if log:
                #print(f">> {ln}   << {resp}")
            if i < len(lines) - 1:
                time.sleep(gap_s)
        # bump the limiter timestamp so next non-burst respects CMD_RATE_HZ
        nonlocal last_cmd_ts
        last_cmd_ts = time.time()
        return resp
    
    #Persistent following if target disappears
    last_target_x = None
    last_target_time = None
    target_persistent_seconds = 3.0
    active_target_id = None           # currently locked track ID
    active_target_last_seen = 0.0     # last time we saw that ID
    
    # --- RealSense setup ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
    #config.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 200)
    profile = pipeline.start(config)
        
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 9)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # >>> VIDEO RECORDER  <<<
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("drive_recording.mp4", fourcc, 30, (COLOR_W, COLOR_H))
    #out_depth = cv2.VideoWriter("depth_recording.mp4", fourcc, 30, (DEPTH_W, DEPTH_H))

    try:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        if depth_sensor.supports(rs.option.laser_power):
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            target = min(rng.max, max(rng.min, rng.max * 0.9))
            depth_sensor.set_option(rs.option.laser_power, target)
    except Exception as e:
        print(f"[WARN] Emitter/laser options not applied: {e}")

    spat, temp, hole = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter(2)
    model = YOLO(YOLO_MODEL)
    
    if not os.path.exists(YUNET_PATH):
        raise FileNotFoundError(f"YUNet model not found at {YUNET_PATH}")

    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError("Your OpenCV version does not support FaceDetectorYN. Requires >= 4.5.5.")

    face_detector = cv2.FaceDetectorYN_create(
        model=YUNET_PATH,
        config="",
        input_size=(COLOR_W, COLOR_H),
        score_threshold=FACE_CONF_THRESHOLD,
        nms_threshold=0.4,
        top_k=5000
    )

    prev_ids = set()
    ever_seen_ids = set()
    
    send_cmd("FL 3") 
    send_cmd("HL 5")   # DRL Headlights ON at 5%
    sound_player = None
        
    print("Starting YOLO + ByteTrack + RealSense. Press 'q' to exit.")
    
    prev_time = time.time() #used to calculate FPS

    try:
        while True:
          try:
            now = time.time()
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            accel = frames.first_or_default(rs.stream.accel)
            if not depth_frame or not color_frame:
                continue

            depth_frame_p = spat.process(depth_frame)
            depth_frame_p = temp.process(depth_frame_p)
            depth_frame_p = hole.process(depth_frame_p)

            color = np.asanyarray(color_frame.get_data())
            depth_u16 = np.asanyarray(depth_frame_p.get_data())
            depth_m = depth_u16.astype(np.float32) * depth_scale
            depth_m[depth_u16 == 0] = np.nan
            
            # For depth video recording RealSense built-in colorizer
            #depth_color = np.asanyarray( colorizer.colorize(depth_frame_p).get_data() )

            Hc, Wc = color.shape[:2]
            Hd, Wd = depth_m.shape
            if (Hc, Wc) != (Hd, Wd):
                print(f"[WARN] Shape mismatch color{(Hc,Wc)} depth{(Hd,Wd)}")

            results = model.track(color, tracker=TRACKER_YAML, persist=True, classes=[0,16], verbose=False)

            frame_detections = []
            vis = color.copy()
                
            if results:
                r0 = results[0]
                vis = r0.plot(labels=True, boxes=True)
  
                if r0.boxes is not None and len(r0.boxes) > 0:
                    for b in r0.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].tolist()

                        dist, (xs, ys, xe, ye) = median_depth_in_bbox(
                            depth_m, (x1, y1, x2, y2), (Hc, Wc), patch_frac=0.25
                        )

                        cls_id = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
                        track_id = None
                        if hasattr(b, "id") and b.id is not None:
                            try:
                                track_id = int(b.id[0].item())
                            except Exception:
                                track_id = int(b.id[0]) if len(b.id) > 0 else None
                        conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])

                        frame_detections.append({
                            "id": track_id,
                            "class_id": cls_id,
                            "distance_m": float(dist) if np.isfinite(dist) else None,
                            "prob": conf,
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
                        })

                        label = f"{dist:.2f} m" if np.isfinite(dist) else "N/A"
                        cv2.rectangle(vis, (xs, ys), (xe, ye), (0, 255, 255), 2)
                        put_label_with_bg(vis, label, xs, ys - 5)


            # Auto-headlight check each frame
            check_top_dark_ratio(color, send_cmd, vis=vis)
            
            #Hazzard detection
            cx = max(0, (COLOR_W) // 2)   # center X reference point
            #Hazzard Center
            pts_c = np.array([
                [cx - 200, COLOR_H - 200],   # top-left
                [cx + 200, COLOR_H - 200],   # top-right
                [cx + 200, COLOR_H-50],         # bottom-right
                [cx - 200, COLOR_H-50],         # bottom-left
            ], dtype=np.int32)
            pts_c = pts_c.reshape((-1, 1, 2))
            hazard_c = detect_obstruction_in_poly(depth_m, pts_c, max_dist_m=0.85, min_pixels=20)
            
            #cliff = detect_cliff_in_poly(depth_m, pts_c)
            #if cliff:
            #   put_label_with_bg(vis, "CLIFF", cx, 400, scale=2, thickness=2)
       
            #Hazzard Left
            pts_l = np.array([
                [cx - 400, COLOR_H - 150],   # top-left
                [cx - 200, COLOR_H - 200],   # top-right
                [cx - 200, COLOR_H-40],         # bottom-right
                [cx - 400, COLOR_H-40],         # bottom-left
            ], dtype=np.int32)
            pts_l = pts_l.reshape((-1, 1, 2))
            hazard_l = detect_obstruction_in_poly(depth_m, pts_l, max_dist_m=.9, min_pixels=20)
            #Hazzard Right
            pts_r = np.array([
                [cx + 200, COLOR_H - 200],   # top-left
                [cx + 400, COLOR_H - 150],   # top-right
                [cx + 400, COLOR_H-40],         # bottom-right
                [cx + 200, COLOR_H-40],         # bottom-left
            ], dtype=np.int32)
            pts_r = pts_r.reshape((-1, 1, 2))  
            hazard_r = detect_obstruction_in_poly(depth_m, pts_r, max_dist_m=0.9, min_pixels=20)              

            
            if hazard_c:
                cv2.polylines(vis, [pts_c], isClosed=True, color=(0, 0, 255), thickness=2)
                #cv2.polylines(depth_color, [pts_c], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                pass
                #cv2.polylines(vis, [pts_c], isClosed=True, color=(80, 200, 80), thickness=1)
                #cv2.polylines(depth_color, [pts_c], isClosed=True, color=(0,255,0), thickness=1)
            if hazard_l:
                cv2.polylines(vis, [pts_l], isClosed=True, color=(0, 0, 255), thickness=2)
                #cv2.polylines(depth_color, [pts_l], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                pass
                #cv2.polylines(vis, [pts_l], isClosed=True, color=(80, 200, 80), thickness=1)
                #cv2.polylines(depth_color, [pts_l], isClosed=True, color=(0,255,0), thickness=1)
            if hazard_r:
                cv2.polylines(vis, [pts_r], isClosed=True, color=(0, 0, 255), thickness=2)
                #cv2.polylines(depth_color, [pts_r], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                pass
                #cv2.polylines(vis, [pts_r], isClosed=True, color=(80, 200, 80), thickness=1)
                #cv2.polylines(depth_color, [pts_r], isClosed=True, color=(0,255,0), thickness=1)

                        
            # ---- find closest person or dog only ----
            person_dets = [
                d for d in frame_detections
                if (d["class_id"] == 0 or d["class_id"] == 16)
                   and d["distance_m"] is not None
                   and d["distance_m"] < 8    # Do not follow peole beyond this distance(meters)
                   and d["prob"] is not None
                   and d["prob"] >= 0.50
            ]
  
            if person_dets:
            
                closest = min(person_dets, key=lambda d: d["distance_m"])
                x1, y1, x2, y2 = closest["x1"], closest["y1"], closest["x2"], closest["y2"]
                dist = closest["distance_m"]
                tid = closest["id"]

                #draw_blue_face_on_target(vis, face_detector, x1, y1, x2, y2)
                
                # draw crosshair for closest person (red lines)
                #cx = int((x1 + x2) / 2)
                #cy = int((y1 + y2) / 2)
                #cv2.line(vis, (cx - 30, cy), (cx - 5, cy), (0, 0, 255), 2)
                #cv2.line(vis, (cx + 5, cy), (cx + 30, cy), (0, 0, 255), 2)
                #cv2.line(vis, (cx, cy - 30), (cx, cy - 5), (0, 0, 255), 2)
                #cv2.line(vis, (cx, cy + 5), (cx, cy + 30), (0, 0, 255), 2)
                
                # center of person in pixels
                cx = 0.5 * (x1 + x2)
                last_target_x = cx
                last_target_time = now
    
                # error in [-1, +1], center is 0
                err = (cx / COLOR_W) * 2.0 - 1.0
                draw_path_lines(vis, err)

                # ---- steering decision logic ----
                if abs(err) <= 0.14:
                    steer_us = STEER_CENTER_US
                else:
                    raw_steer = STEER_CENTER_US + (STEER_SPAN_US * (STEER_K * err))

                    if err < -0.14:
                        # round down to nearest 100
                        steer_us = int((raw_steer // 100) * 100)
                        if hazard_l and not hazard_c:
                            steer_us = 1400
                            err = 0
                    else:
                        # round up to nearest 100
                        steer_us = int(((raw_steer + 99) // 100) * 100)
                        if hazard_r and not hazard_c:
                            steer_us = 1400
                            err = 0
                                            
                # clamp to limits
                steer_us = max(STEER_MIN_US, min(STEER_MAX_US, steer_us))
                put_label_with_bg(vis, f"ST: {steer_us}", 1, 35, scale=.6, thickness=1)
                
                    
                # throttle logic with 0.1 floor and 1.0 cap
                thr = 0.0
                if dist > TARGET_DIST_M:
                    thr = THROTTLE_K * (dist - TARGET_DIST_M)
                    # If we intend to move, clamp to [0.1, 1.0]; else keep 0.0 to stop.
                    if thr > 0.0:
                        thr = max(THROTTLE_MIN, min(thr, 1.0))
                        
                left_thr = thr
                right_thr = thr
                
                # debug HUD for error
                put_label_with_bg(vis, f"thr: {thr:.4f}", 1, 52	, scale=.6, thickness=1)
                put_label_with_bg(vis, f"err: {err:.4f}", 1, 69	, scale=.6, thickness=1)
                
                if thr > 0.0:
                      if not hazard_c:
                           if abs(err) > 0.25 and dist<2.5:
                               # Short Distance. Diffrential torque needed to enhance turn
                               differential_factor= THROTTLE_STEER_K*abs(err)
                               differential_factor = min(differential_factor, 1.0)
                               if err < -0.25:
                                    left_thr = max(THROTTLE_MIN, min(thr*(1-differential_factor), 1.0))
                                    right_thr = max(THROTTLE_MIN, min(thr*(1+differential_factor), 1.0))
                               else:
                                    left_thr = max(THROTTLE_MIN, min(thr*(1+differential_factor), 1.0))
                                    right_thr = max(THROTTLE_MIN, min(thr*(1-differential_factor), 1.0))
                      else:
                          if hazard_l and not hazard_r:
                          	steer_us = STEER_MIN_US
                          	left_thr = -0.1
                          	right_thr = -0.2
                          elif hazard_r and not hazard_l:
                          	steer_us = STEER_MAX_US
                          	left_thr = -0.2
                          	right_thr = -0.1
                          else:
                                # center hazard, stop
                                left_thr = 0.0
                                right_thr = 0.0
                                steer_us = STEER_CENTER_US 
                else:
                     left_thr = 0.0
                     right_thr = 0.0
                     steer_us = STEER_CENTER_US
                
                
                # ramp left toward left_thr
                if left_pwr_state < left_thr:
                    left_pwr_state = min(left_pwr_state + THROTTLE_STEP_UP, left_thr)
                elif left_pwr_state > left_thr:
                    left_pwr_state = max(left_pwr_state - THROTTLE_STEP_DOWN, left_thr)

                # ramp right toward right_thr
                if right_pwr_state < right_thr:
                    right_pwr_state = min(right_pwr_state + THROTTLE_STEP_UP, right_thr)
                elif right_pwr_state > right_thr:
                    right_pwr_state = max(right_pwr_state - THROTTLE_STEP_DOWN, right_thr)
                #Clamp ranges
                left_pwr_state  = max(-1.0, min(left_pwr_state,  1.0))
                right_pwr_state = max(-1.0, min(right_pwr_state, 1.0))
                #Send Control Signals to Servo and Motors
                tk_line = f"TK {left_pwr_state:.3f} {right_pwr_state:.3f}"
                send_burst([f"ST {steer_us}", tk_line], log=True)
                          	
                # HUD: steering 
                steer_pct = np.interp(steer_us, [STEER_MIN_US, STEER_CENTER_US, STEER_MAX_US], [-100, 0, 100])
                if abs(steer_pct) < 1.0:
                    steer_txt = "center 0%"
                elif steer_pct < 0:
                    steer_txt = f"left {abs(steer_pct):.0f}%"
                else:
                    steer_txt = f"right {steer_pct:.0f}%"
                
                # wheel power in percent from thr (same on both wheels here)
                l_pct = int(round(left_pwr_state * 100.0))
                r_pct = int(round(right_pwr_state  * 100.0))
                hud = f"Steer: {steer_txt}   |   L: {l_pct}%  R: {r_pct}%"
                
                # place text at bottom middle
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale, thickness = 0.7, 2
                (tw, th), _ = cv2.getTextSize(hud, font, scale, thickness)
                tx = max(0, (COLOR_W - tw) // 2)
                ty = COLOR_H - 12  # a bit above bottom
                put_label_with_bg(vis, hud, tx, ty, scale=scale, thickness=thickness)

                # Small label near box
                put_label_with_bg(vis, "Follow", x1, max(0, y2))

                # ===== NEW: compute new IDs vs previous frame and trigger =====
                # Collect IDs present this frame (ignore None)
                curr_ids = {d["id"] for d in frame_detections if d["id"] is not None}

                # New this frame vs immediate previous frame
                new_ids_vs_prev = curr_ids - prev_ids

                if new_ids_vs_prev:
                    # Call once per new id
                    for det in frame_detections:
                        if det["id"] in new_ids_vs_prev:
                            on_new_detection(det)

                # Update state for next frame
                prev_ids = curr_ids
                ever_seen_ids |= curr_ids
            else:
                # Target lost. Reset power state 
                left_pwr_state  = 0
                right_pwr_state = 0
                #Try Turn in direction of last target seen until target_persistent_seconds
                if last_target_x is not None and (now -last_target_time)<target_persistent_seconds:
                    if last_target_x > COLOR_W // 2 + 200 and not hazard_c and not hazard_r:
                        print(f"Using cached RIGHT target position: X = {last_target_x} at {last_target_time} / {now}")
                        send_burst([f"ST {STEER_MAX_US}", "TK 0.3 -0.2"], log=True)
                        hud = "Steer: right 100%    |   L: 20%  R: -10%"
                        # place text at bottom middle
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale, thickness = 0.7, 2
                        (tw, th), _ = cv2.getTextSize(hud, font, scale, thickness)
                        tx = max(0, (COLOR_W - tw) // 2)
                        ty = COLOR_H - 12  # a bit above bottom
                        put_label_with_bg(vis, hud, tx, ty, scale=scale, thickness=thickness)
                    elif last_target_x < COLOR_W // 2 - 200 and not hazard_c and not hazard_l:
                        print(f"Using cached LEFT target position: X = {last_target_x} at {last_target_time} / {now}")
                        send_burst([f"ST {STEER_MIN_US}", "TK -0.2 0.3"], log=True)
                        hud = "Steer: left 100%   |   L: -10%  R: 20%"
                        # place text at bottom middle
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale, thickness = 0.7, 2
                        (tw, th), _ = cv2.getTextSize(hud, font, scale, thickness)
                        tx = max(0, (COLOR_W - tw) // 2)
                        ty = COLOR_H - 12  # a bit above bottom
                        put_label_with_bg(vis, hud, tx, ty, scale=scale, thickness=thickness)
                else:
                    # No person found -> STOP car
                    send_burst(["TK 0 0"], log=True)            # cut throttle
                    send_cmd(f"ST {STEER_CENTER_US}")           # recenters steering

                    # Clear current IDs so the next appearance can trigger on_new_detection again
                    prev_ids = set()

                    # HUD: show STOP status at bottom
                    l_pct = r_pct = 0
                    hud = "No person - STOP   |   L: 0%  R: 0%"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale, thickness = 0.7, 2
                    (tw, th), _ = cv2.getTextSize(hud, font, scale, thickness)
                    tx = max(0, (COLOR_W - tw) // 2)
                    ty = COLOR_H - 12
                    put_label_with_bg(vis, hud, tx, ty, scale=scale, thickness=thickness)

            # HUD: show IMU Data on top left
            if accel:
               a = accel.as_motion_frame().get_motion_data()
               ax, ay, az = a.x, a.y, a.z

               g_force = np.sqrt(ax**2 + ay**2 + az**2) / 9.81
               if max_g_force < g_force:
                   max_g_force = g_force
                   
               forces = (f"G:{g_force:.2f}g / {max_g_force:.2f}g")
               put_label_with_bg(vis, forces, 1, 15, scale=scale, thickness=1)
            else:
               forces = "IMU: no data"
               put_label_with_bg(vis, forces, 1, 14, scale=scale, thickness=1)
            
            #Display FPS
            current_time = time.time()
            dt = current_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0
            prev_time = current_time
            FPS = (f"FPS: {fps:.2f}")
            put_label_with_bg(vis, FPS, 400, 15, scale=0.7, thickness=1)

            # Write to view file
            out.write(vis)
            #out_depth.write(depth_color)
            
            # Display cv2
            cv2.imshow("YOLO + ByteTrack + RealSense (Follow person)", vis)
            #cv2.imshow("Depth (colorized)", depth_color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
          
          except Exception as e:
            print(f"[RS ERROR] {e}")

            if "5000" in str(e) or "frame didn't arrive" in str(e).lower():
                # Stop pipeline safely
                try:
                    print("[RS] Stopping  due to frame timeout...")
                    pipeline.stop()
                except:
                    pass

                # Realsense Reset needed
                print("Realsense Reset needed.")
                break
            else:
                    # Some other error: re-raise so you see it
                    raise     
    finally:
        try:
            send_burst(["TK 0 0", "HL 0"], log=True)
            send_cmd(f"ST {STEER_CENTER_US}")           # recenters steering
            time.sleep(0.05)  # 50 ms to ensure TX buffer flushes
            ser.close()
            pipeline.stop()
        except Exception:
            pass
        
        out.release()
        #out_depth.release()
        cv2.destroyAllWindows()
       
if __name__ == "__main__":
    main()

