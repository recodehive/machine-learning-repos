# Virtual Painter Project

## Description
The Virtual Painter is an interactive project where users can draw on the screen by tracking a colored object (e.g., a red pen or green object) via a webcam. As the object moves across the screen, it leaves a virtual trail, creating a painting or drawing effect. This project uses OpenCV to detect the object's color and track its movement, allowing for real-time drawing on the screen.

This fun and engaging project can be used for educational purposes, drawing games, or creative activities by tracking specific color objects and adjusting the canvas features.

## Features
- **Real-time color tracking**: Detect and track an object with a specific color using the HSV color range.
- **Dynamic Drawing**: Draw on the screen by moving the color object in front of the webcam.
- **Customizable canvas**: Modify the color range and adjust the drawing features.
- **Noise Filtering**: Ignores smaller irrelevant contours to prevent noise from interfering with the drawing.

## Dependencies
To run this project, the following Python packages are required:
- OpenCV for image processing:
```pip install opencv-python```

- Numpy for numerical operations:
```pip install numpy```

## How to Run
- Install the required dependencies mentioned above.
- Download or clone the project files:
```git clone https://github.com/your-repo/your-project.git```
- Navigate to the project directory:
```cd your-project-folder```
- Run the script:
```python virtual_painter.py```
- **Use a colored object** (like a red or green pen) in front of the webcam to start drawing. Move the object around to see the trail created on the screen.

## How It Works

The project works by using OpenCV to capture video from the webcam and detect the movement of an object based on its color. The HSV (Hue, Saturation, and Value) color space is used to define a range for detecting specific colors. Once the object is detected, its coordinates are tracked, and a trail is drawn on the canvas.
## Color Detection
The color detection is done using the HSV color range, which separates color (Hue) from intensity (Saturation and Value). The object's color is detected by defining lower and upper bounds in HSV format, which is then used to create a mask to highlight the colored object.
## Tracking and Drawing
Once the object is detected, its position is tracked, and the coordinates are stored in a list. The cv2.line() or cv2.circle() function is then used to draw lines or points on the screen at those coordinates, creating a virtual drawing effect.
## Project Structure
```
- virtual_painter.py   # Main script for the virtual painter
- README.md            # Documentation for the project
```

