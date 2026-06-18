import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque
from tensorflow.keras.models import load_model
import pyttsx3
import threading

# ---------------- LOAD MODEL ----------------
model = load_model("landmark_model.keras")

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# ---------------- TEXT TO SPEECH ----------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def speak_text(text):
    def run():
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()

# ---------------- MEDIAPIPE SETUP ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

buffer = deque(maxlen=25)

sentence = ""
last_output = ""
last_time = 0
cooldown = 2

# ---------------- COLORS ----------------
DARK = (35, 35, 35)
BLACK = (15, 15, 15)
WHITE = (245, 245, 245)
GRAY = (170, 170, 170)
GREEN = (90, 220, 120)
BLUE = (230, 180, 80)
RED = (80, 80, 230)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (960, 640))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = "No hand"
    confidence = 0.0

    # ---------------- HAND DETECTION ----------------
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            min_x = min(x_list)
            min_y = min(y_list)

            data = []

            for lm in hand_landmarks.landmark:
                data.append(lm.x - min_x)
                data.append(lm.y - min_y)

            data = np.array(data).reshape(1, -1)

            prediction = model.predict(data, verbose=0)
            confidence = float(np.max(prediction))
            class_index = int(np.argmax(prediction))

            label = labels.get(class_index, "Unknown")
            buffer.append(label)

            most_common = max(set(buffer), key=buffer.count)
            count = buffer.count(most_common)

            prediction_text = most_common

            current_time = time.time()

            if confidence > 0.80 and count > 8:
                if most_common != last_output and current_time - last_time > cooldown:
                    sentence += " " + most_common
                    sentence = sentence.strip()
                    last_output = most_common
                    last_time = current_time

                    speak_text(most_common)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # ---------------- CLEAN UI OVERLAY ----------------

    # Top bar
    cv2.rectangle(frame, (0, 0), (960, 70), BLACK, -1)
    cv2.putText(
        frame,
        "Sign Language Detection",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        WHITE,
        2
    )

    # Prediction box
    cv2.rectangle(frame, (25, 90), (310, 170), DARK, -1)
    cv2.putText(
        frame,
        "Prediction",
        (45, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GRAY,
        1
    )
    cv2.putText(
        frame,
        prediction_text,
        (45, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        GREEN,
        2
    )

    # Confidence box
    cv2.rectangle(frame, (330, 90), (590, 170), DARK, -1)
    cv2.putText(
        frame,
        "Confidence",
        (350, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GRAY,
        1
    )
    cv2.putText(
        frame,
        f"{int(confidence * 100)}%",
        (350, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        BLUE,
        2
    )

    # Confidence bar
    bar_x, bar_y = 440, 142
    bar_w, bar_h = 120, 10
    fill_w = int(bar_w * confidence)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), GREEN, -1)

    # Status box
    cv2.rectangle(frame, (610, 90), (880, 170), DARK, -1)
    cv2.putText(
        frame,
        "Status",
        (630, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GRAY,
        1
    )

    status = "Detecting" if results.multi_hand_landmarks else "Waiting"
    status_color = GREEN if results.multi_hand_landmarks else BLUE

    cv2.putText(
        frame,
        status,
        (630, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        status_color,
        2
    )

    # Sentence panel
    cv2.rectangle(frame, (0, 510), (960, 590), BLACK, -1)
    cv2.putText(
        frame,
        "Sentence",
        (25, 540),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        GRAY,
        1
    )

    display_sentence = sentence if sentence else "Detected words will appear here..."
    cv2.putText(
        frame,
        display_sentence[-75:],
        (25, 575),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        WHITE,
        2
    )

    # Footer controls
    cv2.rectangle(frame, (0, 590), (960, 640), (20, 20, 20), -1)
    cv2.putText(
        frame,
        "S: Speak sentence   C: Clear   Backspace: Delete last word   Q/Esc: Quit",
        (25, 623),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GRAY,
        1
    )

    cv2.imshow("Sign Language Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    elif key == ord("s"):
        if sentence.strip():
            speak_text(sentence)

    elif key == ord("c"):
        sentence = ""
        last_output = ""
        buffer.clear()

    elif key == 8 or key == 127:
        words = sentence.strip().split()
        sentence = " ".join(words[:-1])
        last_output = ""

cap.release()
cv2.destroyAllWindows()