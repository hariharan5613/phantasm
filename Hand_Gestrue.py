import cv2
import numpy as np
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()


cap = cv2.VideoCapture(0)

x1 = y1 = x2 = y2 = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    results = hands.process(frame)
    multi_hand_landmarks = results.multi_hand_landmarks

    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            draw_utils.draw_landmarks(frame, hand_landmarks)
            onehand_landmark=hand_landmarks.landmark
            for id, lm in enumerate(onehand_landmark):
                x = int(lm.x * frame_width)
                y = int(lm.y * frame_height)

                if id == 8:  
                    x1, y1 = x, y
                    mouse_x = int(screen_width / frame_width * x +30)
                    mouse_y = int(screen_height / frame_height * y +30)
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), 1)
                    pyautogui.moveTo(mouse_x, mouse_y)
                
                if id == 12:  
                    x2, y2 = x, y
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), 1)
        
            dist= y2 - y1
            print(dist)
            if dist< 20:
                pyautogui.click()
                pyautogui.sleep(0.2) 
    cv2.imshow("Hand Click", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
