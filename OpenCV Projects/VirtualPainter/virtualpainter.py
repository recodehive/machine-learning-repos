
#### File: `virtual_painter.py`
```python
import cv2
import numpy as np

# Set up the color ranges for detection (e.g., blue pen)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Initialize a blank canvas to draw on
canvas = np.zeros((480, 640, 3), dtype="uint8")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for detecting the blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours of the detected blue object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assuming it's the pen)
        max_contour = max(contours, key=cv2.contourArea)

        # Get the center of the contour (i.e., the tip of the pen)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        
        # Draw on the canvas at the detected position
        cv2.circle(canvas, center, 5, (255, 0, 0), -1)

    # Merge the original frame with the canvas
    combined = cv2.add(frame, canvas)

    # Display the result
    cv2.imshow('Virtual Painter', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
