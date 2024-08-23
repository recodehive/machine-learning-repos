import mediapipe as mp
import cv2
import numpy as np
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


cap = cv2.VideoCapture(0)

while True:

    rec , frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    results = hands.process(gray_frame)

    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for multihands in results.multi_hand_landmarks:
            for id , lm in enumerate(multihands.landmark):
                h , w , c = frame.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                print(id , cx , cy)

                if id == 4:
                    cv2.circle(frame , (cx , cy) , 15 , (255 , 255 , 9) , cv2.FILLED)

            mpDraw.draw_landmarks(frame , multihands , mpHands.HAND_CONNECTIONS)
    
    cTime  =time.time()
    Fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame , str(int(Fps)) , (10 , 40) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,255 , 0) , 2 )
        

    
    cv2.imshow("webcam" , frame)
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

cap.release()
cv2.destroyAllWindows()

    