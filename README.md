# Herbie Autonomy
<img width="968" height="632" alt="herbie-autonomy-theme-small" src="https://github.com/user-attachments/assets/e735bbe1-dba9-4c72-92e9-4f053ae8a0aa" />

Herbie is an edge-AI autonomous rover built on NVIDIA Jetson Orin Nano and Intel RealSense D435i for real-world person following, depth-aware navigation, and robotics experimentation. The project combines onboard perception, object tracking, motor control, hazard detection, and field-tested visual debugging overlays into a single embedded robotics platform.

This repository contains the Jetson-side autonomy logic and the Arduino-side firmware used to control steering, throttle, and vehicle lighting.

## Overview

Herbie was built as a practical robotics project focused on real-world deployment rather than simulation-only development. The system uses a forward-facing RGB-D camera, onboard inference, tracking, and control logic to detect and follow a person while reacting to nearby hazards and operating under the constraints of embedded compute.

The project emphasizes:

- hardware-software integration
- edge AI perception on Jetson
- real-time control under limited FPS
- depth-aware following behavior
- visual instrumentation for field testing
- real-world validation instead of lab-only assumptions

## Key Features

- YOLO-based person detection running on Jetson
- ByteTrack-based persistent target tracking
- Intel RealSense depth integration for distance estimation
- target lock behavior to keep following the same person ID
- hazard-zone depth checks in front of the rover
- steering and throttle control via Arduino over serial
- rate-limited command transmission for smoother actuation
- auto-headlight brightness logic based on scene lighting
- IMU g-force display in the live HUD
- visual driving path overlay based on steering error
- optional face-blur pipeline for public video privacy
- onboard video recording for review and testing

## System Architecture

Herbie is split into two main layers:

### Jetson / Edge AI Layer
Runs on NVIDIA Jetson Orin Nano and handles:

- color + depth capture from Intel RealSense D435i
- aligned RGB/depth processing
- person detection using YOLO
- person tracking using ByteTrack
- distance estimation from depth patches
- target selection and target persistence
- steering and throttle decisions
- hazard detection from depth polygons
- telemetry overlays and recording

### Arduino Control Layer
Runs on the Arduino and handles:

- steering servo command execution
- motor control
- headlight and flash command execution
- low-level actuation in response to serial commands from Jetson

## Hardware

Current platform components include:

- **NVIDIA Jetson Orin Nano** for onboard AI inference and control logic
- **Intel RealSense D435i** for RGB, depth, and IMU data
- **Arduino-based motor/servo controller**
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
- **ByteTrack**
- **Intel RealSense SDK / pyrealsense2**
- **PySerial**
- **python-vlc**

## Repository Structure

```text
herbie-autonomy/
  README.md
  LICENSE
  requirements.txt
  src/
    jetson/
      herbie_autonomy_follow_person.py
      herbie_autonomy_follow_id.py
    arduino/
      Herbie.ino
  media/
    herbie-front.jpg
    herbie-side.jpg
    tracking-overlay.jpg
