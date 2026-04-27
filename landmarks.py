import cv2
import mediapipe as mp
import math
import json
import asyncio
import websockets
import threading
import os
from ml.features import extract
from ml.classifier import GestureClassifier

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)
stop = False
classifier = GestureClassifier()
gesture_data = {
    "pinch_index": 1.0,
    "pinch_middle": 1.0,
    "rotation": 0.5,
    "gesture": "unknown",
    "gesture_confidence": 0.0,
}

def camera_loop():
    global stop
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pinch_index = 1.0
        pinch_middle = 1.0
        rotation = 0.5
        gesture_label = "unknown"
        gesture_conf = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = list(hand_landmarks.landmark)
                for lm in landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                middle_base = landmarks[9]
                wrist = landmarks[0]

                ref_dx = middle_base.x - wrist.x
                ref_dy = middle_base.y - wrist.y
                hand_size = math.sqrt(ref_dx**2 + ref_dy**2)

                p1_dx = thumb_tip.x - index_tip.x
                p1_dy = thumb_tip.y - index_tip.y
                pinch_index = math.sqrt(p1_dx**2 + p1_dy**2) / hand_size

                p2_dx = thumb_tip.x - middle_tip.x
                p2_dy = thumb_tip.y - middle_tip.y
                pinch_middle = math.sqrt(p2_dx**2 + p2_dy**2) / hand_size

                pinch_index = (pinch_index - 0.2) / (1.0 - 0.2)
                pinch_middle = (pinch_middle - 0.2) / (1.4 - 0.2)
                pinch_index = max(0.0, min(1.0, pinch_index))
                pinch_middle = max(0.0, min(1.0, pinch_middle))

                angle = math.degrees(math.atan2(-ref_dy, ref_dx)) - 90
                rotation = (angle - (-80)) / (90 - (-80))
                rotation = max(0.0, min(1.0, rotation))

                features = extract(hand_landmarks)
                prediction = classifier.predict(features)
                gesture_label = prediction['smoothed']
                gesture_conf = prediction['confidence']

                cv2.putText(frame, f"Pinch Index:  {round(pinch_index, 3)}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Pinch Middle: {round(pinch_middle, 3)}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Rotation:     {round(rotation, 3)}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Gesture: {gesture_label} ({gesture_conf:.2f})",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        gesture_data["pinch_index"] = round(pinch_index, 3)
        gesture_data["pinch_middle"] = round(pinch_middle, 3)
        gesture_data["rotation"] = round(rotation, 3)
        gesture_data["gesture"] = gesture_label
        gesture_data["gesture_confidence"] = round(gesture_conf, 3)

        cv2.imshow("Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
            break

    cap.release()
    cv2.destroyAllWindows()

async def handler(websocket):
    print("Client connected")
    while True:
        data = json.dumps(gesture_data)
        await websocket.send(data)
        await asyncio.sleep(0.033)

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Server running on ws://localhost:8765")
        if classifier.available:
            print(f"Classifier loaded with {len(classifier.labels)} classes: {classifier.labels}")
        else:
            print("No classifier found — run ml/collect.py then ml/train.py to enable predictions.")
        while not stop:
            await asyncio.sleep(0.1)
    os._exit(0)

thread = threading.Thread(target=camera_loop)
thread.daemon = True
thread.start()

asyncio.run(main())
