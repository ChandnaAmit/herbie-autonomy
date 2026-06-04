# Herbie Autonomy
<img width="968" height="633" alt="herbie-autonomy-theme-small" src="https://github.com/user-attachments/assets/cd7cc3ad-1d50-4efa-9b65-72a3647de033" />

Herbie is an edge-AI autonomous rover built on NVIDIA Jetson Orin Nano and RealSense D435i for real-world robotics experimentation, person following, depth-aware navigation, and sidewalk-scale autonomy.

The project started as a hands-on way to learn Python and embedded AI, and has evolved into a field-tested autonomy platform combining onboard perception, depth sensing, semantic path detection, local planning, PlayStation controller takeover, clean/debug video logging, and Arduino-based motor control.

This repository contains the Jetson-side autonomy logic and Arduino-side firmware used to control steering, throttle, and vehicle lighting.

## Overview

Herbie was built as a practical robotics project focused on real-world deployment rather than simulation-only development. The system uses a forward-facing RGB-D camera, onboard inference, tracking, semantic segmentation, depth-based hazard detection, and control logic to operate under real outdoor conditions.

The current autonomy stack supports both person-following behavior and autonomous path driving. In autonomous mode, Herbie uses semantic segmentation to detect drivable path regions, evaluates ribbon-based local planner candidates, and converts the selected path into steering and throttle commands.

The project emphasizes:

- hardware-software integration
- edge AI perception on Jetson
- real-time control under limited FPS
- semantic drivable-path detection
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

## Key Features

- YOLO-based person detection running on Jetson
- ByteTrack-based persistent target tracking
- YOLO semantic segmentation for drivable path detection
- ribbon/band-based local planner for autonomous sidewalk navigation
- RealSense D435i RGB/depth integration
- depth-based distance estimation for person following
- depth hazard-zone checks in front of the rover
- center, left, and right hazard regions
- path confidence logic using planner band coverage
- short bad-frame hold behavior to reduce one-frame planner flicker
- mask cleanup using OpenCV morphology before planner scoring
- PlayStation controller mode switching and manual driving
- autonomous, manual, closest-person follow, and locked-ID follow modes
- steering and throttle control through Arduino over serial
- rate-limited serial command transmission
- auto-headlight brightness logic based on scene lighting
- IMU g-force display in debug HUD
- visual driving path and planner overlays
- optional face-blur pipeline for public video privacy
- clean and debug video recording
- video replay mode for testing autonomy logic on recorded footage

## Data Flywheel

Herbie now records two videos during field testing:

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

When Herbie encounters an edge case, such as glare, shadows, grass boundaries, water-side paths, or imperfect segmentation, the debug video provides the frame number and the clean video provides the corresponding raw frame for labeling.

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

Herbie is split into two main layers:

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
- **YOLO semantic segmentation**
- **ByteTrack**
- **RealSense SDK / pyrealsense2**
- **PySerial**
- **pygame**
- **TensorRT**

## Autonomy Planner

The autonomous path planner uses a semantic segmentation mask to identify drivable space. It then evaluates multiple ribbon-shaped candidate paths across the image.

Each candidate is divided into bands. The planner scores how much of each band lies inside the drivable mask, selects the best candidate, and converts the selected path into steering and throttle commands.

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

Manual takeover is not just a convenience. It is part of the field-testing workflow. When Herbie reaches an edge case, the operator can switch to manual mode, recover the rover, then switch back to autonomous mode within seconds.

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
