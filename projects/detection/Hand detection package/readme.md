# Hand Detection Module

This project is a Python module that utilizes MediaPipe and OpenCV for real-time hand detection and landmark recognition using a webcam. The module detects hand landmarks and highlights specific points, such as the tip of the thumb, on the video feed.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Limitations](#limitations)
## Introduction

This hand detection module is designed for real-time detection and tracking of hand landmarks. It leverages MediaPipe's pre-trained hand detection model and OpenCV for video capture and processing.

## Features

- Real-time hand detection using a webcam.
- Identification and marking of specific hand landmarks (e.g., thumb tip).
- Display of the video feed with overlayed hand landmarks and frames per second (FPS).
- Simple and efficient, with the ability to handle multiple hands.

## Installation

To use this module, ensure you have Python installed along with the required libraries:

```bash
pip install opencv-python
pip install mediapipe
```

## Usage

To run the hand detection module, execute the script:

```bash
python Hand_detection_module.py
```

The script will open your webcam and start detecting hand landmarks in real-time. Press 'x' to exit the application.

## How It Works

1. **Video Capture**: The script captures video from the default webcam using OpenCV.
2. **Frame Processing**: Each frame is converted to RGB and processed by the MediaPipe Hands model to detect hand landmarks.
3. **Landmark Detection**: The detected landmarks are identified and their coordinates are calculated.
4. **Landmark Highlighting**: Specific landmarks (like the thumb tip) are highlighted on the video feed using colored circles.
5. **FPS Calculation**: The script calculates and displays the frames per second (FPS) to monitor performance.

### Key Sections of the Code

- **Hand Landmark Detection**: The code uses `mp.solutions.hands` to detect hand landmarks and `mp.solutions.drawing_utils` to draw them on the video feed.
- **Specific Landmark Highlighting**: The tip of the thumb (landmark ID 4) is highlighted with a filled circle.
- **FPS Display**: The FPS is calculated based on the time difference between consecutive frames and displayed on the video feed.

## Limitations

- **Lighting Conditions**: The accuracy of hand detection can be affected by poor lighting conditions.
- **Single Camera Source**: The script is designed to work with a single webcam source and may require modifications for other video inputs.
- **Dependency on MediaPipe**: This module heavily relies on the MediaPipe library, so any updates or changes in MediaPipe might require adjustments to the code.

