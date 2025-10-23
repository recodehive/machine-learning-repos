import cv2
import mediapipe as mp
import Hand_detection_module as hdm
import numpy as np
import time
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()

vol = volume.GetVolumeRange()


vMin = vol[0]
vMax = vol[1]




########################
wCam ,hCam = 720 ,  620
########################


cap  = cv2.VideoCapture(0)
cap.set(3 , wCam)
cap.set(4,hCam)
pTime = 0
per = 0

delector = hdm.HandDetection(min_detection_confidence= .6)

while True:
    rec , frame = cap.read()

    frame = delector.findHand(frame )
    lmlist = delector.findPosition(frame , draw= False)
    if len(lmlist) != 0:

        print(lmlist[4] , lmlist[8])
        x1, y1 = lmlist[4][1] , lmlist[4][2]
        x2, y2 = lmlist[8][1] , lmlist[8][2]
        cv2.circle(frame , (x1, y1) ,15 , (255, 0 ,255) , -1 )
        cv2.circle(frame , (x2, y2) ,15 , (255, 0 ,255) , -1 )
        cv2.line(frame , (x1, y1) , (x2 , y2) , (255, 0 , 255), 3)
        cx , cy = (x1 + x2)//2 , (y1+y2)//2
        cv2.circle(frame , (cx , cy) ,15 , (255, 0 ,255) , -1 )

        dis = math.hypot(x2-x1 , y2 - y1)
        # print(dis)
        # range of dis = ( 50 , 300)

        finalVol = np.interp(dis , [50 , 280] , [vMin , vMax])
        height = np.interp(dis , [50 , 280] , [400 , 150])
        vol = np.interp(dis , [50 , 280] , [0 , 100])
        volume.SetMasterVolumeLevel(finalVol, None)

        print(finalVol)

        cv2.rectangle(frame , (50 , 150) , (85 , 400) , (0, 255, 0) , 3)

        cv2.rectangle(frame , (50 , int(height)) , (85 , 400) , (0, 256 , 0) , -1)
        cv2.putText(frame , f'{str(int(vol))} %' , (48 , 458) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0, 256 , 0) , 2 )





        if dis < 50:
            cv2.circle(frame , (cx , cy) ,15 , (0, 0 ,255) , -1 )
        
        if dis > 280:
            cv2.circle(frame , (cx , cy) ,15 , (0, 255 ,0) , -1 )






    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame , f'FPS : {str(int(fps))}' , (10 , 40) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0, 255 , 0) , 2 )

    cv2.imshow("webcam" , frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()

