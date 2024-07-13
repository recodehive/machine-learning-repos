import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()

    img = cv2.flip(img, 1)
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)
            finger_fold_status = []
            for tip in finger_tips:
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)

                if lm_list[tip].x < lm_list[tip - 2].x:
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)
            print(x, y)

            x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)
            print(x, y)

            if all(finger_fold_status):
                if lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y:
                    cv2.putText(img, "Peace", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("Peace")

            # Hello
            if lm_list[4].y < lm_list[2].y and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < \
                    lm_list[5].x:
                cv2.putText(img, "HELLO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("HELLO")

            # thank you
            if lm_list[thumb_tip].y < lm_list[1].x and lm_list[0].x < lm_list[3].x:
                print("THANK YOU")
                cv2.putText(img, "THANK YOU", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # I love you
            if lm_list[16].x < lm_list[13].x and lm_list[12].x < lm_list[9].x and lm_list[0].x and lm_list[1].x and \
                    lm_list[2].x and lm_list[3].x and lm_list[4].x and lm_list[5].x and lm_list[6].x and lm_list[
                7].x and lm_list[8].x and lm_list[17].x and lm_list[18].x and lm_list[19].x and lm_list[20].x:
                cv2.putText(img, "ILoveYou", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("ILoveYou")

            # Sorry
            if lm_list[thumb_tip].y < lm_list[20].y and lm_list[5].x and lm_list[6].x and lm_list[7].x and lm_list[
                8].x and lm_list[9].x and lm_list[10].x and lm_list[11].x and lm_list[12].x and lm_list[13].x and \
                    lm_list[14].x and lm_list[15].x and lm_list[16].x and lm_list[17].x and lm_list[18].x and lm_list[
                19].x and lm_list[20].x:
                cv2.putText(img, "SORRY", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                print("SORRY")

            # # Come to Play
            # if lm_list[16].x < lm_list[13].x and lm_list[12].x < lm_list[9].x and lm_list[0].x and lm_list[1].x and \
            #         lm_list[2].x and lm_list[3].x and lm_list[4].x and lm_list[5].x > lm_list[8].x and lm_list[17].x and \
            #         lm_list[18].x and lm_list[19].x and lm_list[20].x:
            #     # cv2.putText(img, "Come to Play", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            #     # print("Come to Play")
            #
            # # You are looking Beautiful
            #
            # if lm_list[12].y < lm_list[10].y and \
            #         lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < \
            #         lm_list[5].x and lm_list[8].y < lm_list[5].x:
            #     # cv2.putText(img, "You are looking Beautiful", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            #     # print("You are looking Beautiful")

        mp_draw.draw_landmarks(img, hand_landmark,
                               mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                               mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                               )

    cv2.imshow("Hand Sign Detection", img)

    cv2.waitKey(1)