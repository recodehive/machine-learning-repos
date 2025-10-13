import cv2
import pickle

# Initialize list to hold parking space coordinates
posList = []

# Load video
cap = cv2.VideoCapture('carPark.mp4')

# Load first frame to select parking spaces
success, img = cap.read()

# Mouse callback function to get parking space positions
def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:  # If left mouse button is clicked
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:  # If right mouse button is clicked, remove the last point
        for i, pos in enumerate(posList):
            if pos[0] < x < pos[0] + 107 and pos[1] < y < pos[1] + 48:  # Within parking space size
                posList.pop(i)

    # Save the parking space positions into a file using pickle
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)

# Show video frame and enable mouse clicks to mark parking positions
while True:
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + 107, pos[1] + 48), (0, 255, 0), 2)  # Draw a rectangle for each parking space
    
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Close video feed
cap.release()
cv2.destroyAllWindows()
