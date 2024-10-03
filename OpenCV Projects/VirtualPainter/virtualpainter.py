import cv2
import numpy as np

# HSV color range for detecting the color object (adjust this range for different colors)
lower_bound = np.array([0, 120, 70])
upper_bound = np.array([10, 255, 255])

# Initialize variables
my_points = []  # List to store points for drawing

# Function to detect color and return the coordinates of the detected object
def find_color(img, lower_bound, upper_bound):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)  # Create mask for specific color

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter by area size to remove noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the object
            return x + w // 2, y

    return None

# Function to draw on canvas based on detected points
def draw_on_canvas(points, img):
    for point in points:
        cv2.circle(img, (point[0], point[1]), 10, (0, 0, 255), cv2.FILLED)  # Draw red circles at each point

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()  # Read frame from webcam
    if not success:
        break

    # Find the object in the current frame
    new_point = find_color(img, lower_bound, upper_bound)

    # If a new point is detected, add it to the list
    if new_point:
        my_points.append(new_point)

    # Draw on the canvas using the points stored
    draw_on_canvas(my_points, img)

    # Display the frame
    cv2.imshow("Virtual Painter", img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
