import argparse
import csv
import os
import time
from pathlib import Path

import sys
from pathlib import Path

import cv2
import mediapipe as mp

if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml.features import extract, FEATURE_DIM
else:
    from .features import extract, FEATURE_DIM

GESTURE_KEYS = {
    ord('1'): 'peace',
    ord('2'): 'thumbs_up',
    ord('3'): 'thumbs_down',
    ord('4'): 'fist',
    ord('5'): 'open_palm',
    ord('6'): 'ok',
    ord('7'): 'point',
    ord('8'): 'rock',
    ord('9'): 'l_sign',
    ord('0'): 'none',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(Path(__file__).parent / 'data' / 'gestures.csv'))
    parser.add_argument('--frames-per-sample', type=int, default=1,
                        help='Number of frames to record per keypress')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_path.exists()
    f = open(out_path, 'a', newline='')
    writer = csv.writer(f)
    if not file_exists:
        header = ['label', 'timestamp'] + [f'f{i}' for i in range(FEATURE_DIM)]
        writer.writerow(header)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    counts = {g: 0 for g in GESTURE_KEYS.values()}
    pending_label = None
    pending_frames_left = 0

    print('Controls:')
    for k, name in GESTURE_KEYS.items():
        print(f"  press '{chr(k)}' -> record {name}")
    print("  press 'q' to quit")
    print(f"Writing to {out_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            feats = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                feats = extract(hand)

            if pending_label and pending_frames_left > 0 and feats is not None:
                writer.writerow([pending_label, time.time()] + feats.tolist())
                counts[pending_label] += 1
                pending_frames_left -= 1
                if pending_frames_left == 0:
                    pending_label = None

            y = 30
            cv2.putText(frame, 'Press 1-9, 0 to label current frame; q quits',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30
            if pending_label:
                cv2.putText(frame, f'Recording {pending_label} ({pending_frames_left} left)',
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y += 30
            for label, c in counts.items():
                cv2.putText(frame, f'{label}: {c}', (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += 20

            cv2.imshow('Gesture Collector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in GESTURE_KEYS:
                pending_label = GESTURE_KEYS[key]
                pending_frames_left = args.frames_per_sample
                f.flush()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        f.close()
        print('\nFinal counts:')
        for label, c in counts.items():
            print(f'  {label}: {c}')


if __name__ == '__main__':
    main()
