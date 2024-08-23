# Gesture Control System

## Overview
The Gesture Control System allows users to control the volume of their computer using hand gestures. This is achieved by detecting the distance between the thumb and index finger using a webcam, and then mapping this distance to a volume level. The system is implemented using OpenCV for video capture, MediaPipe for hand detection, and the Pycaw library for controlling the system audio.

## Features
- **Hand Detection:** Utilizes MediaPipe's hand detection module to detect and track the position of the hand in real-time.
- **Gesture Control:** Calculates the distance between the thumb and index finger and maps this distance to control the system's audio volume.
- **Visual Feedback:** Provides real-time visual feedback of the hand positions and the current volume level on the webcam feed.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- Numpy
- Pycaw
- Comtypes

To install the required libraries, you can use the following pip command:
```bash
pip install opencv-python mediapipe numpy pycaw comtypes
```
## How to Run

1. Ensure that your system has a working webcam.
2. Install the required libraries as mentioned in the Requirements section.
3. Run the Python script:
   ```bash
   python Gesture_Control.py
   ```
4. Use your thumb and index finger to control the volume:
   - Bring them closer to decrease the volume.
   - Move them apart to increase the volume.

5. Press `x` on the keyboard to exit the program.

## Advantages
- **Contactless Control:** Allows users to control volume without any physical contact, making it ideal for environments where hands-free operation is essential.
- **Real-time Operation:** The system operates in real-time, providing immediate feedback and control.

## Limitations
- **Lighting Conditions:** The performance of the hand detection might vary depending on the lighting conditions.
- **Single-Purpose:** The system is designed specifically for volume control; extending it to other applications would require additional development.
