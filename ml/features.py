import numpy as np

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3


def landmarks_to_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def normalize(points):
    wrist = points[0].copy()
    centered = points - wrist

    middle_base = centered[9]
    scale = np.linalg.norm(middle_base[:2])
    if scale < 1e-6:
        scale = 1.0
    scaled = centered / scale

    dx, dy = middle_base[0], middle_base[1]
    theta = np.arctan2(dx, -dy)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t, 0.0],
                    [sin_t,  cos_t, 0.0],
                    [0.0,    0.0,   1.0]], dtype=np.float32)
    rotated = scaled @ rot.T

    return rotated.flatten()


def extract(hand_landmarks):
    pts = landmarks_to_array(hand_landmarks.landmark)
    return normalize(pts)
