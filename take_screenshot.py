import numpy as np # type: ignore
import pyautogui # type: ignore
import imutils # type: ignore
import cv2 # type: ignore
import mediapipe as mp # type: ignore

# Initialize MediaPipe Hand Solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture from webcam
cap = cv2.VideoCapture(0)

# Define finger tips and thumb tip landmarks
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            finger_fold_status = []

            # Check if each finger is folded
            for tip in finger_tips:
                if lm_list[tip].x < lm_list[tip - 2].x and abs(lm_list[tip].y - lm_list[tip - 2].y) > 0.05:
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Check if the thumb is extended (thumb tip is to the right of the thumb base joint)
            thumb_folded = lm_list[thumb_tip].x > lm_list[thumb_tip - 2].x

            # Screenshot only if all fingers are folded and the thumb is extended (not folded)
            if all(finger_fold_status) and not thumb_folded:
                pyautogui.screenshot().show()

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    # Display the image with landmarks
    cv2.imshow("Hand Tracking", img)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
