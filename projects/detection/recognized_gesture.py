import cv2
from flask import Flask, render_template
import cv2
import openai
import mediapipe as mp
import pyttsx3
from langchain import ChatOpenAI

# Initialize OpenAI API
openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize text-to-speech
engine = pyttsx3.init()

# Function to send gesture to LLM for context
def interpret_gesture(gesture_description):
    prompt = f"What does the gesture '{gesture_description}' signify?"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=60
    )
    return response.choices[0].text.strip()

# Function for voice output
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Initialize MediaPipe Hands for gesture tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Here you would add logic to classify the hand gesture
            # For simplicity, let's assume we recognize a "thumbs up"
            recognized_gesture = "thumbs up"
            gesture_meaning = interpret_gesture(recognized_gesture)

            print(f"Recognized Gesture: {recognized_gesture}")
            print(f"Interpreted Meaning: {gesture_meaning}")

            # Voice output
            speak_text(f"Gesture: {recognized_gesture}. Meaning: {gesture_meaning}")

    # Display the frame
    cv2.imshow("Gesture Prediction", frame)

    # Exit with the 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Gesture recognition logic can be linked here to update the UI

if __name__ == '__main__':
    app.run(debug=True)
