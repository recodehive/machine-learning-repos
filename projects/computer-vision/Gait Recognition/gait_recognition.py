import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the pre-trained gait recognition model (or train a new one)
model = KNeighborsClassifier(n_neighbors=3)

# Function to perform background subtraction and silhouette extraction
def extract_silhouette(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply background subtraction
    fgmask = cv2.createBackgroundSubtractorMOG2().apply(gray)
    
    # Threshold to binarize the silhouette
    _, silhouette = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    
    return silhouette

# Function to extract gait features from the silhouette
def extract_gait_features(silhouette):
    # Example: Extract contour area as a feature
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return [cv2.contourArea(largest_contour)]
    return [0]  # Return zero if no valid silhouette is found

# Start capturing video (from webcam or pre-recorded video)
cap = cv2.VideoCapture('walking_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract silhouette from the current frame
    silhouette = extract_silhouette(frame)

    # Extract gait features
    gait_features = extract_gait_features(silhouette)

    # Display the silhouette
    cv2.imshow("Silhouette", silhouette)

    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
