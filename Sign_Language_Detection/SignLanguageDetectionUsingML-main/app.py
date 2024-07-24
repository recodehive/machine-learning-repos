# 1.Importing required modules
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# 2.Loading the trained model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# 3.Creating a list of colors
colors = []
for i in range(0, 20):
    colors.append((245, 117, 16))
print(len(colors))

# 4.Defining a probability visualization function
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


# 5.Setting up variables for gesture recognition:
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

# 6.Initializing the video capture
cap = cv2.VideoCapture(0)

# 7.Configuring MediaPipe hands model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        cropframe = frame[40:400, 0:300]
        # print(frame.shape)
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        # frame=cv2.putText(frame,"Active Region",(75,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,255,2)
        image, results = mediapipe_detection(cropframe, hands)
        # print(results)

        # Draw landmarks
        # draw_styled_landmarks(image, results)
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))

                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]


                # Viz probabilities
                # frame = prob_viz(res, actions, frame, colors,threshold)
        except Exception as e:
            # print(e)q
            pass

        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()