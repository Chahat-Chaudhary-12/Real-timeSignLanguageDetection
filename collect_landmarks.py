import cv2
import mediapipe as mp
import numpy as np
import os

WORD = "pee"   # change word here

SAVE_PATH = f"landmark_data/{WORD}"
os.makedirs(SAVE_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

count = 0
max_samples = 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            min_x, min_y = min(x_list), min(y_list)

            data = []
            for lm in hand_landmarks.landmark:
                data.append(lm.x - min_x)
                data.append(lm.y - min_y)

            np.save(f"{SAVE_PATH}/{count}.npy", data)
            count += 1

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"{WORD} Count: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collect Landmarks", frame)

    if cv2.waitKey(1) == 27 or count >= max_samples:
        break

cap.release()
cv2.destroyAllWindows()