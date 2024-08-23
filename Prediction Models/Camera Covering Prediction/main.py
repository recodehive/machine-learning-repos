import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def predict_camera_covering(frame):
    # Preprocess the frame
    frame = cv2.resize(frame, (128, 128))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0  # Rescale pixel values to [0, 1]

    # Calculate the mean value of the frame
    value = cv2.mean(frame)
    return value

def generate_frames():
    cap = cv2.VideoCapture(0)  # Change to 1 or 2 if multiple cameras are available
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the mean value to determine camera covering
        prediction = predict_camera_covering(frame)
        threshold = 0.33  # Adjust threshold value as needed

        # Display the prediction result on the frame
        if prediction[0] < threshold:
            text = "Covered"
            color = (0, 0, 255)  # Red
        else:
            text = "Uncovered"
            color = (0, 255, 0)  # Green
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the frame
        cv2.imshow('Camera Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_frames()
