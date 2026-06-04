import cv2
import os
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
import serial, time
import pygame

PORT = "/dev/ttyACM0"
BAUD = 115200

TRACKER_YAML = "/home/amit/bytetrack.yaml"
YOLO_MODEL   = "yolo12m.engine"  # TensorRT engine
YUNET_PATH = "/home/amit/yunet.onnx"
FACE_CONF_THRESHOLD = 0.5
FACE_BLUR = False          # True = blur all detected faces in frame for privacy
FACE_BLUR_K = 31          # Gaussian blur kernel size, must be odd. Larger = stronger blur

RECORD_VIDEO = True
SHOW_VIDEO = True
DEBUG_VIEW = False

# 848x480 keeps color==depth after align()
COLOR_W, COLOR_H = 848, 480
DEPTH_W, DEPTH_H = 848, 480

# --- Behavior ---
MODE_FOLLOW_LOCKED = "Locked Follow"
MODE_FOLLOW_CLOSEST = "Person Follow"
MODE_AUTONOMOUS = "Autonomous"
MODE_MANUAL = "Manual"

# --- Settings ---
TARGET_DIST_M   = 1.0
STEER_K         = 1.0 #steering aggressiveness
THROTTLE_K      = 0.5 #Throttle response as target gets further
THROTTLE_STEER_K = 0.8    # Aggressiveness of diffrencial Left right speeds to enhance turn
THROTTLE_STEP_UP = 0.02   #Throttle max increment per frame. Smooth motion
THROTTLE_STEP_DOWN = 0.2 #Throttle max decrease per frame. Smooth motion
STEER_CENTER_US = 1400  #Center
STEER_SPAN_US   = 300
STEER_MIN_US    = 1100  #Max left
STEER_MAX_US    = 1700  #Max right
THROTTLE_MIN = 0.0  # per Arduino: anything below 0.08 is ignored

# throttle command rate limit (to avoid flooding Arduino)
CMD_RATE_HZ     = 30.0

# Joystick for manual mode
USE_CONTROLLER = True
JOYSTICK_DEADZONE = 0.08
LEFT_STICK_THROTTLE_AXIS = 1
RIGHT_STICK_STEER_AXIS = 2
INVERT_THROTTLE_AXIS = True
INVERT_STEER_AXIS = False

BTN_MANUAL = 0
BTN_AUTONOMOUS = 1
BTN_PERSON_FOLLOW = 3
BTN_LOCKED_FOLLOW = 4

# --- Autonomous semantic/ribbon planner ---
USE_VIDEO_INPUT = True
VIDEO_INPUT_PATH = "DHL2_clean480.mp4"

SEMANTIC_MODEL = "best_640_yolo26s-sem.engine"
SEMANTIC_IMG_SIZE = 640

MASK_ALPHA = 0.35
RIBBON_ALPHA = 0.50
DRIVABLE_CLASS_ID = 1

RIBBON_BOTTOM_WIDTH = 360
RIBBON_END_WIDTH = 70
RIBBON_TRAVEL_RATIO = 0.33
RIBBON_BOTTOM_IGNORE_RATIO = 0.02
RIBBON_BANDS = 8
BAND_VALID_COVERAGE = 0.90
RIBBON_CURVE_POWER = 1.75
SHIFT_STEP_X = 35
MAX_SHIFT_X = 175

AUTONOMOUS_ERR_DEADBAND = 0.024  # when steering starts
AUTONOMOUS_ERR_FULL_STEER = 0.055  # when steering reaches full available strength
AUTONOMOUS_MAX_THROTTLE = 0.55
AUTONOMOUS_MIN_GOOD_BANDS = 2

# Comm with Arduino
def open_port():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(1.0)
    ser.reset_input_buffer()
    return ser

# Simple serial helper for one-line commands to the Arduino.
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


def init_controller():
    if not USE_CONTROLLER:
        return None

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("[CTRL] No controller found")
        return None

    controller = pygame.joystick.Joystick(0)
    controller.init()

    print(f"[CTRL] Connected: {controller.get_name()}")
    print(f"[CTRL] Axes: {controller.get_numaxes()} Buttons: {controller.get_numbuttons()} Hats: {controller.get_numhats()}")

    return controller


def poll_controller(controller, current_mode):
    if controller is None:
        return current_mode, STEER_CENTER_US, 0.0, 0.0, 0.0

    try:
        pygame.event.pump()

        prev_buttons = getattr(poll_controller, "_prev_buttons", None)
        curr_buttons = [bool(controller.get_button(i)) for i in range(controller.get_numbuttons())]

        if prev_buttons is None:
            prev_buttons = curr_buttons.copy()

        def pressed_once(button_index):
            if button_index >= len(curr_buttons):
                return False
            return curr_buttons[button_index] and not prev_buttons[button_index]

        steer_axis = controller.get_axis(RIGHT_STICK_STEER_AXIS) if RIGHT_STICK_STEER_AXIS < controller.get_numaxes() else 0.0
        throttle_axis = controller.get_axis(LEFT_STICK_THROTTLE_AXIS) if LEFT_STICK_THROTTLE_AXIS < controller.get_numaxes() else 0.0

        if INVERT_STEER_AXIS:
            steer_axis = -steer_axis

        if INVERT_THROTTLE_AXIS:
            throttle_axis = -throttle_axis

        if abs(steer_axis) < JOYSTICK_DEADZONE:
            steer_axis = 0.0

        if abs(throttle_axis) < JOYSTICK_DEADZONE:
            throttle_axis = 0.0

        steer_axis = max(-1.0, min(1.0, steer_axis))
        throttle_axis = max(-1.0, min(1.0, throttle_axis))

        manual_steer_min = STEER_MIN_US + 100
        manual_steer_max = STEER_MAX_US - 100

        if steer_axis < 0:
            manual_steer_us = STEER_CENTER_US + int((STEER_CENTER_US - manual_steer_min) * steer_axis)
        else:
            manual_steer_us = STEER_CENTER_US + int((manual_steer_max - STEER_CENTER_US) * steer_axis)

        manual_steer_us = max(manual_steer_min, min(manual_steer_max, manual_steer_us))
        manual_thr = throttle_axis

        if pressed_once(BTN_MANUAL):
            current_mode = MODE_MANUAL
        elif pressed_once(BTN_AUTONOMOUS):
            current_mode = MODE_AUTONOMOUS
        elif pressed_once(BTN_PERSON_FOLLOW):
            current_mode = MODE_FOLLOW_CLOSEST
        elif pressed_once(BTN_LOCKED_FOLLOW):
            current_mode = MODE_FOLLOW_LOCKED

        poll_controller._prev_buttons = curr_buttons

        return current_mode, manual_steer_us, manual_thr, steer_axis, throttle_axis

    except Exception as e:
        print(f"[CTRL] Controller poll failed: {e}")
        return current_mode, STEER_CENTER_US, 0.0, 0.0, 0.0


def handle_keyboard(key, current_mode):
    mode_order = [
        MODE_FOLLOW_CLOSEST,
        MODE_AUTONOMOUS,
        MODE_MANUAL,
        MODE_FOLLOW_LOCKED,
    ]

    if key == ord("q"):
        return current_mode, True

    if key == 32:  # Space bar
        if current_mode in mode_order:
            current_index = mode_order.index(current_mode)
            next_index = (current_index + 1) % len(mode_order)
            return mode_order[next_index], False

        return MODE_FOLLOW_CLOSEST, False

    if key == ord("1"):
        return MODE_FOLLOW_CLOSEST, False

    if key == ord("2"):
        return MODE_FOLLOW_LOCKED, False

    if key == ord("3"):
        return MODE_AUTONOMOUS, False

    if key == ord("4"):
        return MODE_MANUAL, False

    return current_mode, False

# Autonomy Functions

def to_numpy(data):
    if hasattr(data, "cpu"):
        return data.cpu().numpy()
    return np.asarray(data)


def get_drivable_mask(result, width, height):
    if not hasattr(result, "semantic_mask") or result.semantic_mask is None:
        return None

    data = to_numpy(result.semantic_mask.data)

    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        else:
            data = np.argmax(data, axis=0)

    class_map = data.astype(np.uint8)

    if class_map.shape[:2] != (height, width):
        class_map = cv2.resize(class_map, (width, height), interpolation=cv2.INTER_NEAREST)

    return class_map == DRIVABLE_CLASS_ID


def draw_mask_overlay(overlay, mask, color, alpha):
    colored = np.zeros_like(overlay)
    colored[mask] = color
    return cv2.addWeighted(overlay, 1.0, colored, alpha, 0)


def interpolate_width(bottom_width, top_width, t):
    return int(bottom_width + (top_width - bottom_width) * t)


def curve_center_x(start_x, shift_x, t, curve_power):
    curved_t = t ** curve_power
    return int(start_x + shift_x * curved_t)


def build_score_mask(width, height, bottom_ignore_ratio):
    score_mask = np.ones((height, width), dtype=bool)
    ignore_bottom_px = int(height * bottom_ignore_ratio)

    if ignore_bottom_px > 0:
        score_mask[height - ignore_bottom_px:height, :] = False

    return score_mask


def build_ribbon_band_mask(width, height, x_bottom, x_top, y_bottom, y_top, band_bottom_width, band_top_width):
    band_mask = np.zeros((height, width), dtype=np.uint8)

    bottom_half = band_bottom_width // 2
    top_half = band_top_width // 2

    points = np.array([
        [x_bottom - bottom_half, y_bottom],
        [x_bottom + bottom_half, y_bottom],
        [x_top + top_half, y_top],
        [x_top - top_half, y_top],
    ], dtype=np.int32)

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    cv2.fillPoly(band_mask, [points], 1)
    return band_mask, points


def build_smooth_ribbon_mask(width, height, start_x, bottom_y, travel_px, bottom_width, end_width, shift_x, curve_power, t_end=1.0, steps=48):
    left_points = []
    right_points = []

    for i in range(steps + 1):
        t = t_end * i / steps
        y = int(bottom_y - travel_px * t)
        x = curve_center_x(start_x, shift_x, t, curve_power)
        ribbon_width = interpolate_width(bottom_width, end_width, t)
        half_width = ribbon_width // 2

        left_points.append([x - half_width, y])
        right_points.append([x + half_width, y])

    points = np.array(left_points + right_points[::-1], dtype=np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)

    return mask.astype(bool), points


def draw_soft_mask_overlay(overlay, mask, color, alpha, blur_size=9):
    mask_float = mask.astype(np.float32)
    mask_float = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    mask_float = np.clip(mask_float, 0.0, 1.0)

    colored = np.zeros_like(overlay, dtype=np.float32)
    colored[:, :] = color

    overlay_float = overlay.astype(np.float32)
    mask_alpha = mask_float[:, :, None] * alpha

    blended = overlay_float * (1.0 - mask_alpha) + colored * mask_alpha
    return blended.astype(np.uint8)


def draw_smooth_ribbon_candidate(overlay, candidate):
    first_failed_band = candidate["first_failed_band"]

    if first_failed_band is None:
        visible_band_count = RIBBON_BANDS
    else:
        visible_band_count = max(0, first_failed_band - 1)

    if visible_band_count == 0:
        return overlay

    t_end = visible_band_count / RIBBON_BANDS

    smooth_mask, _ = build_smooth_ribbon_mask(
        COLOR_W, COLOR_H,
        bottom_center_x, bottom_y, ribbon_travel_px,
        RIBBON_BOTTOM_WIDTH, RIBBON_END_WIDTH,
        candidate["shift_x"], RIBBON_CURVE_POWER,
        t_end=t_end,
        steps=48
    )

    overlay = draw_soft_mask_overlay(overlay, smooth_mask, (0, 255, 0), RIBBON_ALPHA, blur_size=15)

    centerline_points = get_smooth_centerline_points(candidate["shift_x"], t_end)
    overlay = draw_centerline(overlay, centerline_points)

    return overlay


def build_ribbon_bands(width, height, start_x, bottom_y, travel_px, bottom_width, end_width, band_count, shift_x, curve_power):
    bands = []
    top_y = max(0, bottom_y - travel_px)

    for i in range(band_count):
        t0 = i / band_count
        t1 = (i + 1) / band_count

        y0 = int(bottom_y - travel_px * t0)
        y1 = int(bottom_y - travel_px * t1)

        x0 = curve_center_x(start_x, shift_x, t0, curve_power)
        x1 = curve_center_x(start_x, shift_x, t1, curve_power)

        width0 = interpolate_width(bottom_width, end_width, t0)
        width1 = interpolate_width(bottom_width, end_width, t1)

        band_mask, band_points = build_ribbon_band_mask(width, height, x0, x1, y0, y1, width0, width1)

        bands.append({
            "index": i + 1,
            "mask": band_mask,
            "points": band_points,
            "x_bottom": x0,
            "x_top": x1,
            "y_bottom": y0,
            "y_top": y1,
            "width_bottom": width0,
            "width_top": width1,
        })

    top_x = curve_center_x(start_x, shift_x, 1.0, curve_power)
    return bands, top_x, top_y


def prepare_band_scoring(bands, score_mask):
    for band in bands:
        band_pixels = band["mask"] > 0
        scored_pixels = band_pixels & score_mask

        band["pixels"] = band_pixels
        band["scored_pixels"] = scored_pixels
        band["area"] = np.sum(scored_pixels)

    return bands


def score_band(band, drivable_mask):
    band_area = band["area"]
    band_inside = np.sum(band["scored_pixels"] & drivable_mask)
    band_outside = band_area - band_inside
    band_coverage = band_inside / band_area if band_area > 0 else 1.0
    band_is_valid = band_coverage >= BAND_VALID_COVERAGE

    return {
        "pixels": band["pixels"],
        "area": band_area,
        "inside": band_inside,
        "outside": band_outside,
        "coverage": band_coverage,
        "is_valid": band_is_valid,
    }


def build_shift_candidates(step_x, max_shift_x):
    shifts = [0]

    for shift in range(step_x, max_shift_x + 1, step_x):
        shifts.append(shift)     # right bias
        shifts.append(-shift)

    return shifts


def build_candidate_band_cache(width, height, start_x, bottom_y, travel_px, score_mask):
    candidate_band_cache = {}

    for shift_x in SHIFT_CANDIDATES:
        bands, top_x, top_y = build_ribbon_bands(
            width, height, start_x, bottom_y, travel_px,
            RIBBON_BOTTOM_WIDTH, RIBBON_END_WIDTH, RIBBON_BANDS,
            shift_x, RIBBON_CURVE_POWER
        )

        bands = prepare_band_scoring(bands, score_mask)

        candidate_band_cache[shift_x] = {
            "bands": bands,
            "top_x": top_x,
            "top_y": top_y,
        }

    return candidate_band_cache


def candidate_name_from_shift(shift_x):
    if shift_x == 0:
        return "straight"

    if shift_x > 0:
        return f"right_{shift_x}"

    return f"left_{abs(shift_x)}"


def evaluate_ribbon_candidate(name, shift_x, candidate_band_cache, drivable_mask):
    cached_candidate = candidate_band_cache[shift_x]
    bands = cached_candidate["bands"]

    band_scores = []
    total_ribbon_pixels = 0
    total_inside_pixels = 0
    first_failed_band = None
    failed_band_count = 0

    for i, band in enumerate(bands):
        band_score = score_band(band, drivable_mask)
        band_scores.append(band_score)

        total_ribbon_pixels += band_score["area"]
        total_inside_pixels += band_score["inside"]

        if not band_score["is_valid"]:
            first_failed_band = band["index"]
            failed_band_count = RIBBON_BANDS - first_failed_band + 1

            for remaining_band in bands[i + 1:]:
                total_ribbon_pixels += remaining_band["area"]

            break

    is_fully_valid = first_failed_band is None

    if is_fully_valid:
        failed_band_count = 0

    total_coverage = total_inside_pixels / total_ribbon_pixels if total_ribbon_pixels > 0 else 0.0

    return {
        "name": name,
        "shift_x": shift_x,
        "bands": bands,
        "band_scores": band_scores,
        "top_x": cached_candidate["top_x"],
        "top_y": cached_candidate["top_y"],
        "total_coverage": total_coverage,
        "failed_band_count": failed_band_count,
        "first_failed_band": first_failed_band,
        "is_fully_valid": is_fully_valid,
    }


def get_candidate_rank(candidate):
    good_band_count = RIBBON_BANDS - candidate["failed_band_count"]
    first_failed_band = candidate["first_failed_band"] if candidate["first_failed_band"] is not None else RIBBON_BANDS + 1
    right_bias = 1 if candidate["shift_x"] > 0 else 0

    return (
        good_band_count,
        candidate["total_coverage"],
        first_failed_band,
        -abs(candidate["shift_x"]),
        right_bias,
    )


def find_best_ribbon_candidate(candidate_band_cache, drivable_mask):
    best_candidate = None

    for shift_x in SHIFT_CANDIDATES:
        candidate = evaluate_ribbon_candidate(
            name=candidate_name_from_shift(shift_x),
            shift_x=shift_x,
            candidate_band_cache=candidate_band_cache,
            drivable_mask=drivable_mask,
        )

        if best_candidate is None or get_candidate_rank(candidate) > get_candidate_rank(best_candidate):
            best_candidate = candidate

    if best_candidate["is_fully_valid"]:
        best_candidate["planner_status"] = "valid"
    else:
        best_candidate["planner_status"] = "best_partial"

    return best_candidate


def draw_ribbon_band(overlay, band, band_score):
    band_color = (0, 255, 0) if band_score["is_valid"] else (0, 0, 255)

    overlay = draw_mask_overlay(overlay, band_score["pixels"], band_color, RIBBON_ALPHA)

    label_x = int(band["x_bottom"] + band["width_bottom"] // 2 + 10)
    label_y = int((band["y_bottom"] + band["y_top"]) / 2)
    cv2.putText(overlay, f"{band['index']}:{band_score['coverage'] * 100:.0f}%", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, band_color, 1)

    return overlay


def draw_ribbon_candidate(overlay, candidate):
    visible_bands = []

    for band, band_score in zip(candidate["bands"], candidate["band_scores"]):
        overlay = draw_ribbon_band(overlay, band, band_score)
        visible_bands.append(band)

        if not band_score["is_valid"]:
            break

    centerline_points = get_centerline_points(visible_bands)
    overlay = draw_centerline(overlay, centerline_points)

    return overlay


def get_centerline_points(bands):
    points = []

    if not bands:
        return points

    first_band = bands[0]
    points.append((first_band["x_bottom"], first_band["y_bottom"]))

    for band in bands:
        points.append((band["x_top"], band["y_top"]))

    return points


def get_smooth_centerline_points(shift_x, t_end, steps=24):
    points = []

    for i in range(steps + 1):
        t = t_end * i / steps
        x = curve_center_x(bottom_center_x, shift_x, t, RIBBON_CURVE_POWER)
        y = int(bottom_y - ribbon_travel_px * t)
        points.append((x, y))

    return points


def draw_centerline(overlay, centerline_points):
    if len(centerline_points) < 2:
        return overlay

    points = np.array(centerline_points, dtype=np.int32)
    cv2.polylines(overlay, [points], False, (255, 255, 255), 1)

    return overlay

def compute_planner_signals(candidate):
    good_band_count = RIBBON_BANDS - candidate["failed_band_count"]
    throttle_signal = good_band_count / RIBBON_BANDS

    # Weighted centerline steering estimate:
    # 25% band 2, 50% band 3, 25% band 4
    steering_weights = [
        (2, 0.5),
        (3, 0.5),
    ]

    weighted_x = 0.0
    total_weight = 0.0

    for band_index, weight in steering_weights:
        if band_index <= len(candidate["bands"]):
            band = candidate["bands"][band_index - 1]

            # Use x_top because the white centerline is drawn through each band's top center.
            band_cx = band["x_top"]

            weighted_x += band_cx * weight
            total_weight += weight

    if total_weight > 0:
        cx = weighted_x / total_weight
    else:
        cx = bottom_center_x

    err = (cx / COLOR_W) * 2.0 - 1.0

    return {
        "good_band_count": good_band_count,
        "throttle_signal": throttle_signal,
        "cx": cx,
        "err": err,
    }


def run_autonomous_planner(color, semantic_model):
    results = semantic_model.predict(source=color, imgsz=SEMANTIC_IMG_SIZE, device=0, verbose=False)
    result = results[0]

    drivable_mask = get_drivable_mask(result, COLOR_W, COLOR_H)

    if drivable_mask is None:
        return None, None, None, "No Drivable Mask"

    # Clean semantic mask before ribbon scoring.
    # CLOSE fills small black holes inside the drivable path.
    # OPEN removes tiny isolated blue/drivable speckles.
    mask_u8 = drivable_mask.astype(np.uint8)

    close_kernel = np.ones((7, 7), np.uint8)
    open_kernel = np.ones((3, 3), np.uint8)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)

    drivable_mask = mask_u8.astype(bool)

    selected_candidate = find_best_ribbon_candidate(candidate_band_cache, drivable_mask)
    planner_signals = compute_planner_signals(selected_candidate)

    return drivable_mask, selected_candidate, planner_signals, "Autonomous Path"


# Auto-headlight logic:
# Compare brightness of top vs bottom of frame.
# If the top gets too dark relative to the bottom, switch headlights to ON otherwise dimmed Daytime Running Lights.
def check_top_dark_ratio(bgr_frame, send_cmd,
                         top_frac=0.2, bottom_frac=0.2,
                         on_ratio=0.55,    # turn HL 20 on if top is <60% of bottom
                         off_ratio=0.68,   # go back to DRL HL 5 if top is >85% of bottom
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
        #put_label_with_bg(vis, f"{light_status} {ratio:.2f} [{on_ratio:.2f}/{off_ratio:.2f}]", 600, 14, scale=0.5, thickness=1)
        put_label_with_bg(vis, f"{light_status}", 600, 14, scale=0.5, thickness=1)

    return ratio, st["hl_on"]

def put_label_with_bg(img, text, x, y, scale=0.6, thickness=1, color=(0, 255, 0), show_bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    pad = 4

    if show_bg:
        x1 = max(0, x - pad)
        y1 = max(0, y - th - pad)
        x2 = min(img.shape[1] - 1, x + tw + pad)
        y2 = min(img.shape[0] - 1, y + bl + pad)

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# Search outward from a target pixel until a valid depth value is found.
# Useful when the exact center pixel is missing or invalid in the depth map.
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


# Build a small center patch inside a detection box for depth sampling.
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


# Measure object distance using the median depth from the bbox center patch.
# Falls back to a nearby valid depth if the patch contains no usable values.
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


# Check whether a hazard polygon contains enough close depth pixels
# to count as a real obstruction in front of the vehicle.
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
        return True

    # Not enough close pixels → treat as no obstruction
    return False


def check_and_draw_hazards(depth_m, vis):
    cx = COLOR_W // 2

    pts_c = np.array([
        [cx - 200, COLOR_H - 200],
        [cx + 200, COLOR_H - 200],
        [cx + 200, COLOR_H - 70],
        [cx - 200, COLOR_H - 70],
    ], dtype=np.int32).reshape((-1, 1, 2))

    pts_l = np.array([
        [cx - 400, COLOR_H - 150],
        [cx - 200, COLOR_H - 200],
        [cx - 200, COLOR_H - 40],
        [cx - 400, COLOR_H - 40],
    ], dtype=np.int32).reshape((-1, 1, 2))

    pts_r = np.array([
        [cx + 200, COLOR_H - 200],
        [cx + 400, COLOR_H - 150],
        [cx + 400, COLOR_H - 40],
        [cx + 200, COLOR_H - 40],
    ], dtype=np.int32).reshape((-1, 1, 2))

    hazard_c = detect_obstruction_in_poly(depth_m, pts_c, max_dist_m=0.80, min_pixels=800)
    hazard_l = detect_obstruction_in_poly(depth_m, pts_l, max_dist_m=0.90, min_pixels=400)
    hazard_r = detect_obstruction_in_poly(depth_m, pts_r, max_dist_m=0.90, min_pixels=400)

    if hazard_c:
        cv2.polylines(vis, [pts_c], isClosed=True, color=(0, 0, 255), thickness=2)

    if hazard_l:
        cv2.polylines(vis, [pts_l], isClosed=True, color=(0, 0, 255), thickness=2)

    if hazard_r:
        cv2.polylines(vis, [pts_r], isClosed=True, color=(0, 0, 255), thickness=2)

    return hazard_c, hazard_l, hazard_r


#Blur Faces for Privacy
def blur_all_faces(vis, face_detector, blur_k=31):
    """
    Detect faces in the full frame and blur each detected face.
    Runs only when FACE_BLUR is enabled.
    """
    H, W = vis.shape[:2]

    result = face_detector.detect(vis)
    faces = result[1]

    if faces is None or len(faces) == 0:
        return

    if blur_k % 2 == 0:
        blur_k += 1

    for f in faces:
        fx, fy, fw, fh = f[:4].astype(int)

        x1 = max(0, fx)
        y1 = max(0, fy)
        x2 = min(W, fx + fw)
        y2 = min(H, fy + fh)

        if x2 <= x1 or y2 <= y1:
            continue

        face_roi = vis[y1:y2, x1:x2]

        if face_roi.size == 0:
            continue

        vis[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (blur_k, blur_k), 0)
    
  
# Draw arc path based on steering error
def draw_path_lines(vis, err, color=(80, 200, 80), thickness=2):
    h, w = vis.shape[:2]

    # where the path begins near the car
    y_bottom = h - 1

    # where the path fades into distance
    y_top = h - 100

    # width of the path near the car and far away
    half_width_bottom = 180
    half_width_top = 30

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


# Displays Steering arrow and power on left and right motors
def draw_steering_wheel_and_power(vis, steer_pct, left_pwr, right_pwr):
    h, w = vis.shape[:2]

    cx = w // 2
    base_y = h - 40

    disp_pct = steer_pct
    angle_deg = (disp_pct / 100.0) * 90.0
    ang = np.deg2rad(angle_deg)

    spoke_len = 45
    sx = int(cx + spoke_len * np.sin(ang))
    sy = int(base_y + 10 - spoke_len * np.cos(ang))
    cv2.arrowedLine(vis, (cx, base_y + 10), (sx, sy), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.22)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 1

    left_txt = f"{int(round(left_pwr * 100))}%"
    state_txt = f"{abs(disp_pct):.0f}%"
    right_txt = f"{int(round(right_pwr * 100))}%"

    (tw, th), _ = cv2.getTextSize(left_txt, font, scale, thickness)
    put_label_with_bg(vis, left_txt, cx - 65 - tw // 2, base_y + 30, scale=scale, thickness=thickness)

    (tw, th), _ = cv2.getTextSize(state_txt, font, scale, thickness)
    put_label_with_bg(vis, state_txt, cx - tw // 2, base_y + 30, scale=scale, thickness=thickness)

    (tw, th), _ = cv2.getTextSize(right_txt, font, scale, thickness)
    put_label_with_bg(vis, right_txt, cx + 65 - tw // 2, base_y + 30, scale=scale, thickness=thickness)


# Custom action when a brand new ID appears in a frame
def on_new_detection(det):
    tid = det["id"]
    cls_id = det["class_id"]
    dist = det["distance_m"]
    print(f"[NEW] id={tid} class={cls_id} dist={dist}")


def read_realsense_frame(pipeline, align, spat, temp, hole, depth_scale):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    accel = frames.first_or_default(rs.stream.accel)

    if not depth_frame or not color_frame:
        return None, None, None

    depth_frame_p = spat.process(depth_frame)
    depth_frame_p = temp.process(depth_frame_p)
    if hole is not None:
        depth_frame_p = hole.process(depth_frame_p)

    color = np.asanyarray(color_frame.get_data())
    depth_u16 = np.asanyarray(depth_frame_p.get_data())

    depth_m = depth_u16.astype(np.float32) * depth_scale
    depth_m[depth_u16 == 0] = np.nan

    Hc, Wc = color.shape[:2]
    Hd, Wd = depth_m.shape

    if (Hc, Wc) != (Hd, Wd):
        print(f"[WARN] Shape mismatch color{(Hc, Wc)} depth{(Hd, Wd)}")

    return color, depth_m, accel


def read_video_frame(cap):
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

        if not ret:
            return None, None, None

    if frame.shape[1] != COLOR_W or frame.shape[0] != COLOR_H:
        frame = cv2.resize(frame, (COLOR_W, COLOR_H), interpolation=cv2.INTER_AREA)

    depth_m = np.full((COLOR_H, COLOR_W), np.nan, dtype=np.float32)
    accel = None

    return frame, depth_m, accel


def detect_people(color, depth_m, model, drive_mode):
    Hc, Wc = color.shape[:2]

    if drive_mode == MODE_FOLLOW_LOCKED:
        results = model.track(color, tracker=TRACKER_YAML, persist=True, classes=[0], verbose=False)
    else:
        results = model.predict(color, classes=[0], verbose=False)

    frame_detections = []
    vis = color.copy()

    if results:
        r0 = results[0]
        vis = r0.plot(labels=False, boxes=True)

        if r0.boxes is not None and len(r0.boxes) > 0:
            for b in r0.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()

                dist, (xs, ys, xe, ye) = median_depth_in_bbox(
                    depth_m, (x1, y1, x2, y2), (Hc, Wc), patch_frac=0.25
                )

                cls_id = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])

                track_id = None
                if drive_mode == MODE_FOLLOW_LOCKED and hasattr(b, "id") and b.id is not None:
                    try:
                        track_id = int(b.id[0].item())
                    except Exception:
                        track_id = int(b.id[0]) if len(b.id) > 0 else None

                conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])

                if conf < 0.35:
                    continue

                frame_detections.append({
                    "id": track_id,
                    "class_id": cls_id,
                    "distance_m": float(dist) if np.isfinite(dist) else None,
                    "prob": conf,
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
                })

                dist_ft = dist * 3.28084 if np.isfinite(dist) else float("nan")
                label = f"{dist_ft:.1f} ft" if np.isfinite(dist_ft) else "N/A"

                cv2.rectangle(vis, (xs, ys), (xe, ye), (0, 255, 255), 2)
                put_label_with_bg(vis, label, xs, ys - 5)

    return frame_detections, vis


def select_person_target(frame_detections, drive_mode, active_target_id, active_target_last_seen, now, target_persistent_seconds):
    person_dets = [
        d for d in frame_detections
        if d["class_id"] == 0
           and d["distance_m"] is not None
           and d["distance_m"] < 8
           and d["prob"] is not None
           and d["prob"] >= 0.40
    ]

    target_det = None

    if not person_dets:
        return None, active_target_id, active_target_last_seen

    if drive_mode == MODE_FOLLOW_CLOSEST:
        target_det = min(person_dets, key=lambda d: d["distance_m"])
        active_target_id = None
        active_target_last_seen = 0.0
        return target_det, active_target_id, active_target_last_seen

    if active_target_id is not None:
        for d in person_dets:
            if d["id"] == active_target_id:
                target_det = d
                active_target_last_seen = now
                break

    if target_det is None and active_target_id is not None:
        if (now - active_target_last_seen) > target_persistent_seconds:
            active_target_id = None

    if target_det is None and active_target_id is None:
        reacquire_candidates = [d for d in person_dets if d["id"] is not None]

        if reacquire_candidates:
            target_det = min(reacquire_candidates, key=lambda d: d["distance_m"])
            active_target_id = target_det["id"]
            active_target_last_seen = now
            print(f"[LOCK] New target id={active_target_id}")

    return target_det, active_target_id, active_target_last_seen

# Cache Setup for Autonomy
SHIFT_CANDIDATES = build_shift_candidates(SHIFT_STEP_X, MAX_SHIFT_X)

bottom_center_x = COLOR_W // 2
bottom_y = COLOR_H - 1
ribbon_travel_px = int(COLOR_H * RIBBON_TRAVEL_RATIO)

score_mask = build_score_mask(COLOR_W, COLOR_H, RIBBON_BOTTOM_IGNORE_RATIO)

candidate_band_cache = build_candidate_band_cache(
    COLOR_W, COLOR_H,
    bottom_center_x, bottom_y, ribbon_travel_px,
    score_mask
)
############### MAIN ##############
# Main runtime loop:
# - reads aligned color + depth from RealSense
# - runs YOLO + ByteTrack person tracking
# - estimates target distance from depth
# - computes steering/throttle commands
# - draws debug HUD overlays
# - optionally blurs all faces for privacy
# - writes video and displays live output
###################################
def main():
    drive_mode = MODE_AUTONOMOUS if USE_VIDEO_INPUT else MODE_FOLLOW_CLOSEST
    # open Arduino serial once

    ser = open_port()
    # simple rate limiter for commands
    last_cmd_ts = 0.0
    min_dt = 1.0 / CMD_RATE_HZ

    max_g_force = 0.0
    left_pwr_state = 0.0
    right_pwr_state = 0.0

    controller = init_controller()
    manual_steer_us = STEER_CENTER_US
    manual_thr = 0.0
    manual_steer_axis = 0.0
    manual_throttle_axis = 0.0

    # Rate-limited single command sender.
    def send_cmd(line: str):
        nonlocal last_cmd_ts
        now = time.time()
        if now - last_cmd_ts < min_dt:
            return
        resp = send_line(ser, line)
        #print(f">> {line}  << {resp}")
        last_cmd_ts = now
    
    # For back-to-back commands that should be sent together
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
    
    # Persistent target-following state.
    # Keeps following the same track ID for a short time even if detection drops briefly.
    last_target_x = None
    last_target_time = None
    target_persistent_seconds = 3.0
    active_target_id = None           # currently locked track ID
    active_target_last_seen = 0.0     # last time we saw that ID

    #Continue for X frames if bad frame
    bad_path_frames = 0
    PATH_BAD_FRAME_LIMIT = 3
    last_autonomous_steer_us = STEER_CENTER_US

    pipeline = None
    cap = None

    if USE_VIDEO_INPUT:
        cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {VIDEO_INPUT_PATH}")

        depth_scale = None
        align = None
        spat, temp, hole = None, None, None
    else:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
        profile = pipeline.start(config)

        align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        try:
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)

            if depth_sensor.supports(rs.option.laser_power):
                rng = depth_sensor.get_option_range(rs.option.laser_power)
                target = rng.min + (rng.max - rng.min) * 0.25
                depth_sensor.set_option(rs.option.laser_power, target)

        except Exception as e:
            print(f"[WARN] Emitter/laser options not applied: {e}")

        spat = rs.spatial_filter()
        temp = rs.temporal_filter()
        hole = None
        #hole = s.hole_filling_filter(2)

    # >>> VIDEO RECORDER  <<<
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    RECORD_FPS = 7
    out = cv2.VideoWriter("drive_recording_debug.mp4", fourcc, RECORD_FPS, (COLOR_W, COLOR_H))
    clean_out = cv2.VideoWriter("drive_recording_clean.mp4", fourcc, RECORD_FPS, (COLOR_W, COLOR_H))
    frame_idx = 0

    model = YOLO(YOLO_MODEL, task="detect")
    semantic_model = YOLO(SEMANTIC_MODEL, task="semantic")

    face_detector = None

    if FACE_BLUR:
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
    
    send_cmd("FL 3")   # Flash Headlights 3 times
    send_cmd("HL 5")   # DRL Headlights ON at 5%

    if USE_VIDEO_INPUT:
        print(f"Starting YOLO + Autonomous Planner from video: {VIDEO_INPUT_PATH}. Press 'q' to exit.")
    else:
        print("Starting YOLO + ByteTrack + RealSense. Press 'q' to exit.")
    
    prev_time = time.time() #used to calculate FPS

    try:
        # Per-frame pipeline:
        # acquire frames -> run perception -> choose target -> compute controls
        # -> draw HUD -> blur faces if enabled -> record/display frame
        while True:
          try:
            now = time.time()

            if USE_VIDEO_INPUT:
                color, depth_m, accel = read_video_frame(cap)
            else:
                color, depth_m, accel = read_realsense_frame(pipeline, align, spat, temp, hole, depth_scale)

            if color is None:
                continue

            clean_frame = color.copy()
            frame_idx += 1

            #Poll Joystick
            drive_mode, manual_steer_us, manual_thr, manual_steer_axis, manual_throttle_axis = poll_controller(controller, drive_mode)


            if drive_mode in [MODE_FOLLOW_CLOSEST, MODE_FOLLOW_LOCKED]:
                frame_detections, vis = detect_people(color, depth_m, model, drive_mode)
            else:
                frame_detections = []
                vis = color.copy()
            status_text = ""

            # Auto-headlight check each frame
            light_ratio, headlights_on = check_top_dark_ratio(color, send_cmd, vis=None)
            light_status = "Lights: ON" if headlights_on else "Lights: DRL"

            # Hazard detection
            if USE_VIDEO_INPUT:
                hazard_c, hazard_l, hazard_r = False, False, False
            else:
                hazard_c, hazard_l, hazard_r = check_and_draw_hazards(depth_m, vis)

            if drive_mode in [MODE_FOLLOW_CLOSEST, MODE_FOLLOW_LOCKED]:
                # ---- find closest person ----
                target_det, active_target_id, active_target_last_seen = select_person_target(
                    frame_detections,
                    drive_mode,
                    active_target_id,
                    active_target_last_seen,
                    now,
                    target_persistent_seconds
                )

                # Active follow mode:
                # steer and throttle toward the locked target while respecting hazard zones.
                if target_det is not None:

                    x1, y1, x2, y2 = target_det["x1"], target_det["y1"], target_det["x2"], target_det["y2"]
                    dist = target_det["distance_m"]
                    tid = target_det["id"]

                    # center of person in pixels
                    cx = 0.5 * (x1 + x2)
                    last_target_x = cx
                    last_target_time = now

                    # error in [-1, +1], center is 0
                    err = (cx / COLOR_W) * 2.0 - 1.0
                    draw_path_lines(vis, err)

                    # throttle logic with 0.1 floor and 1.0 cap
                    thr = 0.0
                    if dist > TARGET_DIST_M:
                        thr = THROTTLE_K * (dist - TARGET_DIST_M)
                        # If we intend to move, clamp to [0.1, 1.0]; else keep 0.0 to stop.
                        if thr > 0.0:
                            thr = max(THROTTLE_MIN, min(thr, 1.0))

                    left_thr = thr
                    right_thr = thr

                    # Limit steering based on speed to prevent over-steer
                    avg_pwr_state = 0.5 * (abs(left_pwr_state) + abs(right_pwr_state))
                    if avg_pwr_state <= 0.25:
                        effective_steer_span = STEER_SPAN_US  # 1100..1700
                    elif avg_pwr_state <= 0.375:
                        effective_steer_span = STEER_SPAN_US - 100  # 1200..1600
                    else:
                        effective_steer_span = STEER_SPAN_US - 200  # 1300..1500

                    # ---- steering decision logic ----
                    if abs(err) <= 0.14:
                        steer_us = STEER_CENTER_US
                    else:
                        raw_steer = STEER_CENTER_US + (effective_steer_span * (STEER_K * err))

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

                    if DEBUG_VIEW:
                        put_label_with_bg(vis, f"ST: {steer_us} SP:{int(effective_steer_span)}", 10, 105, scale=.55,
                                          thickness=1)
                        put_label_with_bg(vis, f"thr: {thr:.4f}", 10, 128, scale=.55, thickness=1)
                        put_label_with_bg(vis, f"err: {err:.4f}", 10, 151, scale=.55, thickness=1)

                    if thr > 0.0:
                        if not hazard_c:
                            if abs(err) > 0.25 and dist < 1.7:
                                # Short Distance. Diffrential torque needed to enhance turn
                                differential_factor = THROTTLE_STEER_K * abs(err)
                                differential_factor = min(differential_factor, 1.0)
                                if err < -0.25:
                                    left_thr = max(THROTTLE_MIN, min(thr * (1 - differential_factor), 1.0))
                                    right_thr = max(THROTTLE_MIN, min(thr * (1 + differential_factor), 1.0))
                                else:
                                    left_thr = max(THROTTLE_MIN, min(thr * (1 + differential_factor), 1.0))
                                    right_thr = max(THROTTLE_MIN, min(thr * (1 - differential_factor), 1.0))
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

                    # steering / wheel-power consistency fix
                    # left turn: steer_us < center -> right wheel should not be slower than left
                    if steer_us < STEER_CENTER_US and left_pwr_state > right_pwr_state:
                        left_pwr_state = right_pwr_state

                    # right turn: steer_us > center -> left wheel should not be slower than right
                    elif steer_us > STEER_CENTER_US and right_pwr_state > left_pwr_state:
                        right_pwr_state = left_pwr_state

                    # Clamp ranges
                    left_pwr_state = max(-1.0, min(left_pwr_state, 1.0))
                    right_pwr_state = max(-1.0, min(right_pwr_state, 1.0))
                    # Send Control Signals to Servo and Motors
                    tk_line = f"TK {left_pwr_state:.3f} {right_pwr_state:.3f}"
                    send_burst([f"ST {steer_us}", tk_line], log=True)

                    # HUD: steering
                    steer_pct = np.interp(
                        steer_us,
                        [STEER_MIN_US, STEER_CENTER_US, STEER_MAX_US],
                        [-100, 0, 100]
                    )
                    draw_steering_wheel_and_power(vis, steer_pct, left_pwr_state, right_pwr_state)

                    # Small label near box
                    if drive_mode == MODE_FOLLOW_CLOSEST:
                        put_label_with_bg(vis, "Person", x1, max(0, y2), scale=0.55, thickness=1, show_bg=True)
                        status_text = "Following Person"
                    else:
                        put_label_with_bg(vis, f"ID {tid}", x1, max(0, y2), scale=0.55, thickness=1, show_bg=True)
                        status_text = f"Following ID {tid}"

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

                # Lost-target behavior:
                # briefly search in the last known direction, then stop and clear the lock.
                else:
                    # Locked target not currently visible
                    left_pwr_state = 0
                    right_pwr_state = 0

                    # Keep trying old direction only while lock timeout has not expired
                    should_search_last_direction = (
                            last_target_x is not None
                            and last_target_time is not None
                            and (now - last_target_time) <= target_persistent_seconds
                            and (
                                    drive_mode == MODE_FOLLOW_CLOSEST
                                    or (
                                            active_target_id is not None
                                            and (now - active_target_last_seen) <= target_persistent_seconds
                                    )
                            )
                    )

                    if should_search_last_direction:

                        if last_target_x > COLOR_W // 2 + 200 and not hazard_c and not hazard_r:
                            print(
                                f"Using cached RIGHT target position: X = {last_target_x} at {last_target_time} / {now}")
                            send_burst([f"ST {STEER_MAX_US}", "TK 0.3 0.1"], log=True)
                            status_text = "SEARCHING RIGHT"
                            draw_steering_wheel_and_power(vis, 100, 0.3, 0.1)
                        elif last_target_x < COLOR_W // 2 - 200 and not hazard_c and not hazard_l:
                            print(
                                f"Using cached LEFT target position: X = {last_target_x} at {last_target_time} / {now}")
                            send_burst([f"ST {STEER_MIN_US}", "TK 0.1 0.3"], log=True)
                            status_text = "SEARCHING LEFT"
                            draw_steering_wheel_and_power(vis, -100, 0.1, 0.3)
                        else:
                            send_burst(["TK 0 0"], log=True)
                            send_cmd(f"ST {STEER_CENTER_US}")
                            status_text = "Target Lost - Stopped"
                            draw_steering_wheel_and_power(vis, 0, 0.0, 0.0)
                    else:
                        # Lock expired or no usable last target position -> STOP and release lock
                        send_burst(["TK 0 0"], log=True)
                        send_cmd(f"ST {STEER_CENTER_US}")

                        active_target_id = None
                        prev_ids = set()

                        status_text = "No Person Selected"
                        draw_steering_wheel_and_power(vis, 0, 0.0, 0.0)

            elif drive_mode == MODE_AUTONOMOUS:
                drivable_mask, selected_candidate, planner_signals, planner_status = run_autonomous_planner(color, semantic_model)

                if planner_signals is None:
                    left_pwr_state = 0.0
                    right_pwr_state = 0.0
                    steer_us = STEER_CENTER_US

                    send_burst(["TK 0 0"], log=True)
                    send_cmd(f"ST {STEER_CENTER_US}")

                    status_text = planner_status
                    draw_steering_wheel_and_power(vis, 0, 0.0, 0.0)

                else:
                    vis = draw_mask_overlay(vis, drivable_mask, (255, 0, 0), MASK_ALPHA)

                    err = planner_signals["err"]
                    throttle_signal = planner_signals["throttle_signal"]
                    good_band_count = planner_signals["good_band_count"]

                    if abs(err) <= AUTONOMOUS_ERR_DEADBAND:
                        drive_err = 0.0
                    else:
                        sign = 1.0 if err > 0 else -1.0
                        drive_err = sign * ((abs(err) - AUTONOMOUS_ERR_DEADBAND) / (
                                    AUTONOMOUS_ERR_FULL_STEER - AUTONOMOUS_ERR_DEADBAND))
                        drive_err = max(-1.0, min(1.0, drive_err))

                    path_good = good_band_count >= AUTONOMOUS_MIN_GOOD_BANDS

                    if path_good:
                        bad_path_frames = 0
                        thr = min(throttle_signal, AUTONOMOUS_MAX_THROTTLE)
                        left_thr = thr
                        right_thr = thr
                        path_hold_active = False
                    else:
                        bad_path_frames += 1

                        if bad_path_frames < PATH_BAD_FRAME_LIMIT:
                            # One or two bad mask frames should not stop the car.
                            # Keep current motor target and previous steering briefly.
                            thr = 0.5 * (left_pwr_state + right_pwr_state)
                            left_thr = left_pwr_state
                            right_thr = right_pwr_state
                            steer_us = last_autonomous_steer_us
                            path_hold_active = True
                        else:
                            # Repeated bad mask frames: ramp down instead of instant hard stop.
                            thr = 0.0
                            left_thr = 0.0
                            right_thr = 0.0
                            steer_us = STEER_CENTER_US
                            path_hold_active = False

                    # Limit steering based on speed to prevent over-steer
                    avg_pwr_state = 0.5 * (abs(left_pwr_state) + abs(right_pwr_state))

                    if avg_pwr_state <= 0.25:
                        effective_steer_span = STEER_SPAN_US - 25
                    elif avg_pwr_state <= 0.55:
                        effective_steer_span = STEER_SPAN_US - 50
                    else:
                        effective_steer_span = STEER_SPAN_US - 100

                    # Steering decision logic
                    if not path_hold_active:
                        if thr <= 0.0 or abs(drive_err) <= 0.01:
                            steer_us = STEER_CENTER_US
                        else:
                            raw_steer = STEER_CENTER_US + (effective_steer_span * (STEER_K * drive_err))
                            steer_us = int(round(raw_steer))

                            if drive_err < 0 and hazard_l and not hazard_c:
                                steer_us = STEER_CENTER_US

                            if drive_err > 0 and hazard_r and not hazard_c:
                                steer_us = STEER_CENTER_US

                        steer_us = max(STEER_MIN_US, min(STEER_MAX_US, steer_us))
                        last_autonomous_steer_us = steer_us

                    # Center hazard stop
                    if hazard_c:
                        left_thr = 0.0
                        right_thr = 0.0
                        left_pwr_state = 0.0
                        right_pwr_state = 0.0
                        steer_us = STEER_CENTER_US

                    # Ramp left toward left_thr
                    if left_pwr_state < left_thr:
                        left_pwr_state = min(left_pwr_state + THROTTLE_STEP_UP, left_thr)
                    elif left_pwr_state > left_thr:
                        left_pwr_state = max(left_pwr_state - THROTTLE_STEP_DOWN, left_thr)

                    # Ramp right toward right_thr
                    if right_pwr_state < right_thr:
                        right_pwr_state = min(right_pwr_state + THROTTLE_STEP_UP, right_thr)
                    elif right_pwr_state > right_thr:
                        right_pwr_state = max(right_pwr_state - THROTTLE_STEP_DOWN, right_thr)

                    # Clamp ranges
                    left_pwr_state = max(-1.0, min(left_pwr_state, 1.0))
                    right_pwr_state = max(-1.0, min(right_pwr_state, 1.0))

                    tk_line = f"TK {left_pwr_state:.3f} {right_pwr_state:.3f}"
                    send_burst([f"ST {steer_us}", tk_line], log=True)

                    if DEBUG_VIEW:
                        vis = draw_ribbon_candidate(vis, selected_candidate)
                    else:
                        vis = draw_smooth_ribbon_candidate(vis, selected_candidate)

                    if path_good:
                        status_text = "Autonomous Path"
                    elif bad_path_frames < PATH_BAD_FRAME_LIMIT:
                        status_text = f"Path Hold {bad_path_frames}/{PATH_BAD_FRAME_LIMIT}"
                    else:
                        status_text = "Searching for Path"

                    if DEBUG_VIEW:
                        put_label_with_bg(vis, f"ST: {steer_us} SP:{int(effective_steer_span)}", 10, 105, scale=.55, thickness=1)
                        put_label_with_bg(vis, f"bands: {good_band_count}/{RIBBON_BANDS}", 10, 128, scale=.55,thickness=1)
                        put_label_with_bg(vis, f"thr: {thr:.4f}", 10, 151, scale=.55, thickness=1)
                        put_label_with_bg(vis, f"err: {err:.4f}", 10, 174, scale=.55, thickness=1)
                        put_label_with_bg(vis, f"drive_err: {drive_err:.4f}", 10, 197, scale=.55, thickness=1)

                    steer_pct = np.interp(
                        steer_us,
                        [STEER_MIN_US, STEER_CENTER_US, STEER_MAX_US],
                        [-100, 0, 100]
                    )

                    draw_steering_wheel_and_power(vis, steer_pct, left_pwr_state, right_pwr_state)

            elif drive_mode == MODE_MANUAL:
                steer_us = manual_steer_us

                left_pwr_state = manual_thr
                right_pwr_state = manual_thr

                left_pwr_state = max(-1.0, min(left_pwr_state, 1.0))
                right_pwr_state = max(-1.0, min(right_pwr_state, 1.0))

                tk_line = f"TK {left_pwr_state:.3f} {right_pwr_state:.3f}"
                send_burst([f"ST {steer_us}", tk_line], log=True)

                steer_pct = np.interp(
                    steer_us,
                    [STEER_MIN_US, STEER_CENTER_US, STEER_MAX_US],
                    [-100, 0, 100]
                )

                status_text = "Manual Control"

                if DEBUG_VIEW:
                    put_label_with_bg(vis, f"ST: {steer_us}", 10, 105, scale=.55, thickness=1)
                    put_label_with_bg(vis, f"thr: {manual_thr:.4f}", 10, 128, scale=.55, thickness=1)
                    put_label_with_bg(vis, f"steer axis: {manual_steer_axis:.3f}", 10, 151, scale=.55, thickness=1)
                    put_label_with_bg(vis, f"thr axis: {manual_throttle_axis:.3f}", 10, 174, scale=.55, thickness=1)

                draw_steering_wheel_and_power(vis, steer_pct, left_pwr_state, right_pwr_state)

            # ---------------- HUD ----------------
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (vis.shape[1], 65), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.22, vis, 0.78, 0, vis)

            # Measure FPS
            current_time = time.time()
            dt = current_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0
            prev_time = current_time

            # Top-left: mode and lights
            put_label_with_bg(vis, f"Mode: {drive_mode}", 10, 24, scale=0.55, thickness=1)
            put_label_with_bg(vis, light_status, 10, 48, scale=0.55, thickness=1)

            # Top-center: main status
            if status_text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale, thickness = 0.58, 1
                (tw, th), _ = cv2.getTextSize(status_text, font, scale, thickness)
                tx = max(0, (COLOR_W - tw) // 2)
                ty = 28
                put_label_with_bg(vis, status_text, tx, ty, scale=scale, thickness=thickness)

            # Bottom-left: FPS and Frame number
            put_label_with_bg(vis, f"FPS: {fps:.1f}", 10, COLOR_H - 12, scale=0.55, thickness=1)
            put_label_with_bg(vis, f"Frame: {frame_idx}", COLOR_W - 150, COLOR_H - 12, scale=0.55, thickness=1)

            # Debug-only IMU
            if DEBUG_VIEW:
                if accel:
                    a = accel.as_motion_frame().get_motion_data()
                    ax, ay, az = a.x, a.y, a.z

                    g_force = np.sqrt(ax ** 2 + ay ** 2 + az ** 2) / 9.81
                    if max_g_force < g_force:
                        max_g_force = g_force

                    forces = f"G:{g_force:.2f}g / {max_g_force:.2f}g"
                    put_label_with_bg(vis, forces, 10, 72, scale=0.55, thickness=1)
                else:
                    put_label_with_bg(vis, "IMU: no data", 10, 72, scale=0.55, thickness=1)
            
            # Optional privacy feature:
            # detect and blur all visible faces in the frame before saving/displaying it.
            if FACE_BLUR and face_detector is not None:
                blur_all_faces(vis, face_detector, blur_k=FACE_BLUR_K)
                put_label_with_bg(vis, f"FaceBlur:{FACE_BLUR}", 700, 32, scale=0.5, thickness=1)

            # Write to view file
            if RECORD_VIDEO:
                out.write(vis)
                clean_out.write(clean_frame)
            
            # Display cv2
            if SHOW_VIDEO:
                cv2.imshow("YOLO + ByteTrack + RealSense (Follow person)", vis)

            key = cv2.waitKey(1) & 0xFF
            drive_mode, should_quit = handle_keyboard(key, drive_mode)

            if should_quit:
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
    # Leave hardware in a safe state on exit:
    # stop motors, center steering, turn off headlights, release camera/video resources.
    finally:
        try:
            send_burst(["TK 0 0", "HL 0"], log=True)
            send_cmd(f"ST {STEER_CENTER_US}")           # recenters steering
            time.sleep(0.05)  # 50 ms to ensure TX buffer flushes
            ser.close()

            if pipeline is not None:
                pipeline.stop()
            if cap is not None:
                cap.release()
        except Exception:
            pass

        out.release()
        clean_out.release()
        cv2.destroyAllWindows()
       
if __name__ == "__main__":
    main()
