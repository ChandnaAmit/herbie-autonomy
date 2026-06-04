# Herbie Autonomy

<img width="968" height="633" alt="herbie-autonomy-theme-small" src="https://github.com/user-attachments/assets/cd7cc3ad-1d50-4efa-9b65-72a3647de033" />

Herbie is an edge-AI autonomous rover built on NVIDIA Jetson Orin Nano and RealSense D435i for real-world robotics experimentation, person following, depth-aware navigation, and sidewalk-scale autonomy.

Herbie combines onboard perception, depth sensing, custom semantic path detection, local planning, PlayStation controller takeover, clean/debug video logging, video replay testing, and Arduino-based motor control into a field-tested embedded robotics platform.

This repository contains the Jetson-side autonomy logic and Arduino-side firmware used to control steering, throttle, and vehicle lighting.

## Demo Videos

- [Outdoor Autonomous Mode field test](https://youtu.be/tP_Axe6zLL8?si=sUzR0vX0LdcxIyvi)
- [Outdoor Person-Following Mode field test](https://youtu.be/ONJ6ncm3Dvs?si=gB_UKVLDzsgjgXbW)

## Overview

Herbie was built as a practical robotics project focused on real-world deployment rather than simulation-only development. The system uses a forward-facing RGB-D camera, onboard inference, tracking, semantic segmentation, depth-based hazard detection, and control logic to operate under real outdoor conditions.

The current autonomy stack supports both person-following behavior and autonomous path driving. In autonomous mode, Herbie uses a custom-trained semantic segmentation model to detect drivable path regions, evaluates ribbon-based local planner candidates, and converts the selected path into steering and throttle commands.

The project emphasizes:

- hardware-software integration
- edge AI perception on Jetson
- real-time control under limited FPS
- custom semantic drivable-path detection
- depth-aware hazard checks
- person following and target tracking
- joystick-based manual takeover
- field-test logging and replay
- real-world validation instead of lab-only assumptions
- data flywheel development through clean video collection and retraining

## Current Operating Modes

Herbie currently supports four operating modes:

### 1. Person Follow Mode

Herbie detects people using YOLO and follows the closest valid person based on RealSense depth distance.

### 2. Locked Follow Mode

Herbie uses ByteTrack to keep following the same tracked person ID, rather than switching to the nearest person every frame.

### 3. Autonomous Mode

Herbie uses YOLO semantic segmentation to detect drivable path regions and a ribbon/band-based local planner to select a safe local path. The planner converts the selected ribbon candidate into steering and throttle commands.

### 4. Manual Mode

Herbie can be driven manually using a PlayStation controller. This is critical for field testing because the operator can recover the rover from edge cases, reposition it, and switch back into autonomy without touching the vehicle or using a keyboard.

## Custom Perception Model Training

Herbie's autonomous path-driving stack uses a custom drivable-path dataset built around the conditions the rover actually sees from its low outdoor camera angle.

The dataset was intentionally organized around a Herbie-specific sidewalk autonomy taxonomy:

- path geometry: straight sidewalks, turns, forks, path-width changes, crosswalks, and sidewalk transitions
- lighting conditions: shadows, glare, direct sun, dusk lighting, and high-contrast scenes
- surface boundaries: grass edges, mulch, gravel, water-side paths, broken edges, and imperfect path boundaries
- deployment perspective: low-mounted camera views from the rover rather than car-height street-scene imagery

The first training set included roughly 400 annotated sidewalk images. The goal was not generic road or sidewalk segmentation. The goal was to teach the model to identify the local drivable region from Herbie's operating viewpoint, where grass, path edges, water, shadows, glare, and surface transitions can all affect planning.

Earlier model iterations included YOLO11s segmentation experiments. The current autonomous stack uses a YOLO semantic segmentation model exported to TensorRT for low-latency Jetson deployment.

The perception pipeline is designed around a continuous data flywheel: field test, identify failure frames, label new edge cases, retrain, export to TensorRT, replay on recorded video, and test again on the rover.

## Key Features

- Custom-trained YOLO semantic segmentation for drivable-path detection
- TensorRT-exported perception model for Jetson deployment
- YOLO-based person detection and ByteTrack persistent target tracking
- RealSense D435i RGB/depth integration for distance estimation and hazard checks
- Ribbon/band-based local planner designed for low-camera sidewalk autonomy
- Four operating modes: autonomous, manual, closest-person follow, and locked-ID follow
- PlayStation controller takeover and one-button mode switching
- Jetson-to-Arduino control split for perception/planning vs. low-level actuation
- Clean/debug video recording for field-test review and dataset improvement
- Video replay mode for testing planner changes on recorded footage

## Data Flywheel

Herbie records two videos during field testing:

### Debug Video

The debug video includes overlays such as:

- current mode
- semantic path mask
- selected planner ribbon
- steering/throttle display
- hazard boxes
- FPS
- frame number

### Clean Video

The clean video records the raw camera feed with no overlays. This is used for future dataset creation and retraining.

When Herbie encounters an edge case, such as glare, shadows, grass boundaries, water-side paths, forks, or imperfect segmentation, the debug video provides the frame number and the clean video provides the corresponding raw frame for labeling.

This turns every field test into training data for the next model iteration.

## Video Replay Mode

Herbie can switch from live RealSense input to recorded video input. This allows the same autonomy code to run on previously recorded clean video.

This is useful for:

- testing planner changes at home
- comparing parameter changes on the same footage
- replaying edge cases
- tuning ribbon width, curve power, steering response, and path confidence
- reducing the number of outdoor field-test iterations needed

## System Architecture

Herbie is intentionally split into two compute/control layers.

The Jetson handles high-level autonomy work: camera input, AI inference, tracking, semantic segmentation, local planning, depth hazard checks, video logging, and operator interface. The Arduino handles deterministic low-level actuation: steering servo commands, motor throttle commands, lighting, and command execution.

This separation keeps perception and planning decoupled from low-level vehicle control. AI inference timing can vary frame to frame, but the Arduino remains responsible for executing simple actuation commands and leaving the vehicle in a safe state when the Jetson exits or loses control.

### Jetson / Edge AI Layer

Runs on NVIDIA Jetson Orin Nano and handles:

- RGB and depth capture from RealSense D435i
- aligned color/depth processing
- person detection using YOLO
- person tracking using ByteTrack
- semantic drivable-path segmentation
- mask cleanup before planner scoring
- ribbon/band-based local path planning
- path confidence and bad-frame handling
- distance estimation from depth patches
- target selection and target persistence
- steering and throttle decisions
- hazard detection from depth polygons
- PlayStation controller input
- mode switching
- telemetry overlays
- clean and debug video recording
- video replay testing

### Arduino Control Layer

Runs on the Arduino and handles:

- steering servo command execution
- motor throttle control
- differential motor commands
- headlight and flash command execution
- low-level actuation in response to serial commands from Jetson

## Hardware

Current platform components include:

- **NVIDIA Jetson Orin Nano** for onboard AI inference and control logic
- **RealSense D435i** for RGB, depth, and IMU data
- **Arduino-based motor/servo controller**
- **PlayStation controller** for manual takeover and mode switching
- **servo steering system**
- **dual-motor drivetrain**
- **vehicle lighting / headlights**
- **battery-powered mobile rover platform**

## Software Stack

Main software components used in this project include:

- **Python**
- **OpenCV**
- **NumPy**
- **Ultralytics YOLO**
- **Custom YOLO semantic segmentation**
- **TensorRT model export/deployment**
- **ByteTrack**
- **RealSense SDK / pyrealsense2**
- **PySerial**
- **pygame**

## Autonomy Planner

The autonomous path planner uses a semantic segmentation mask to identify drivable space. It then evaluates multiple ribbon-shaped candidate paths across the image.

Each candidate is divided into bands. The planner scores how much of each band lies inside the drivable mask, selects the best candidate, and converts the selected path into steering and throttle commands.

The ribbon planner was chosen because Herbie operates from a low, forward-facing camera where the immediate question is not global route planning, but whether a vehicle-sized corridor can fit through the currently visible drivable region. Earlier sector-style scoring was useful for simple steering, but it was less expressive for sidewalk curves, forks, and path-width changes. The ribbon approach lets the planner evaluate multiple curved local path candidates, divide each candidate into bands, and require high drivable-mask coverage before committing to motion.

Several planner choices are deliberately conservative for outdoor testing. Each ribbon band must meet a high coverage threshold, the semantic mask is cleaned with morphology before scoring, steering is estimated from near/mid lookahead bands instead of the closest pixels, and short bad-frame hold logic prevents one noisy segmentation frame from immediately stopping the rover. This makes the planner more useful for real field testing, where glare, shadows, grass boundaries, and mask flicker are expected.

The planner includes:

- configurable ribbon width
- configurable ribbon travel distance
- curved path candidates
- left/right shift candidates
- band coverage scoring
- steering estimate from selected path bands
- throttle scaling from valid path confidence
- bad-frame hold behavior for temporary segmentation dropouts
- smooth visual overlay for selected candidate path

## Safety and Field Testing

Herbie includes multiple safety and testing mechanisms:

- PlayStation controller manual takeover
- one-button mode switching
- depth-based center hazard stop
- side hazard steering suppression
- manual recovery from autonomy edge cases
- clean video recording for dataset improvement
- debug video recording for failure analysis
- video replay mode before outdoor retesting
- safe-state shutdown behavior for motors, steering, and headlights
- optional YuNet-based face detection and Gaussian face blur for public video privacy

Manual takeover is not just a convenience. It is part of the field-testing workflow. When Herbie reaches an edge case, the operator can switch to manual mode, recover the rover, then switch back to autonomous mode within seconds.

## Why This Project Matters

Herbie is a small robot, but the development loop mirrors the same core challenges found in larger autonomy programs: perception, planning, safety, control, operator takeover, field testing, logging, replay, failure analysis, and continuous model improvement.

Every outdoor field test generates useful failure data: glare frames, shadow transitions, grass boundaries, water-side paths, forks, mask dropouts, and planner hesitation points. Those failures feed directly into the next model version, planner tuning pass, or field-test procedure. The goal is a rover that gets measurably better with each deployment cycle.

## Repository Structure

```text
herbie-autonomy/
  README.md
  LICENSE
  requirements.txt
  src/
    jetson/
      herbie_autonomy.py
    arduino/
      Herbie.ino
  media/
    herbie-front.jpg
    herbie-side.jpg
    tracking-overlay.jpg
    autonomous-path-overlay.jpg
