import cv2
import time
import mediapipe as mp

# mode = False , maxHands = 2 , detectionCon = 0.5 , TrackCon = 0.5
class HandDetection:
    def __init__(self , min_detection_confidence = 0.5):
        # self.mode = mode
        # self.maxHand = maxHands
        self.min_detection_confidence = min_detection_confidence
        # self.TrackCon = TrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands( self.min_detection_confidence )
        self.mpDraw = mp.solutions.drawing_utils


    def findHand(self , frame , flag = True):
        RGB_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(RGB_frame)

        # print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for multihands in self.results.multi_hand_landmarks:
                if flag:
                    self.mpDraw.draw_landmarks(frame , multihands , self.mpHands.HAND_CONNECTIONS)

        return frame
    
    def findPosition(self , frame , handno = 0 , draw = True):

        lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handno]

            for id , lm in enumerate(myHand.landmark):
                h , w , c = frame.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                # print(id , cx , cy)
                lmList.append([id , cx , cy])

                if draw:
        
                    cv2.circle(frame , (cx , cy) , 7 , (255 , 0 , 9) , cv2.FILLED)

        return lmList

    
def main():
    pTime = 0
    cTime = 0


    cap = cv2.VideoCapture(0)
    
    detector = HandDetection()


    while True:

        rec , frame = cap.read()

        frame = detector.findHand(frame)
        lmlist = detector.findPosition(frame)
        
        if len(lmlist) != 0:
            print(lmlist[4])
        
        cTime  =time.time()
        Fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(frame , str(int(Fps)) , (10 , 40) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,255 , 0) , 2 )
        

    
        cv2.imshow("webcam" , frame)
        if cv2.waitKey(1) & 0xFF == ord("x"):
         break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()