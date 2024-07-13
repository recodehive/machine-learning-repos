import cv2
import mediapipe as mp

# Load the MediaPipe Handpose model


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Define the landmarks for each sign
hello_landmarks = [0, 1, 2, 3]
world_landmarks = [0, 9, 10, 11]
help_landmarks = [0, 1, 2, 5]


while True:
    # Capture a frame from the camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Detect landmarks on the hand in the frame using the MediaPipe Handpose model
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_hands.process(image)
    image.flags.writeable = True
    if results.multi_hand_landmarks:
        # Extract the landmarks for the first hand
        hand_landmarks = results.multi_hand_landmarks[0].landmark

        # Extract the x, y, z coordinates for each landmark
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.append(landmark.x)
            landmarks.append(landmark.y)
            landmarks.append(landmark.z)

        # Recognize the sign based on the landmark positions
        if set(hello_landmarks).issubset(set(range(len(landmarks)))):
            print("Hello!")
        elif set(world_landmarks).issubset(set(range(len(landmarks)))):
            print("World!")
        elif set(help_landmarks).issubset(set(range(len(landmarks)))):
            print("Help!")

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()