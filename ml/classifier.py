import json
from collections import deque
from pathlib import Path

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

MODEL_DIR = Path(__file__).parent / 'models'


class GestureClassifier:
    def __init__(self, model_path=None, labels_path=None, smoothing_window=5):
        model_path = Path(model_path or MODEL_DIR / 'gesture_model.joblib')
        labels_path = Path(labels_path or MODEL_DIR / 'labels.json')

        self.available = (joblib is not None and
                          model_path.exists() and labels_path.exists())
        if not self.available:
            self.model = None
            self.labels = []
            return

        self.model = joblib.load(model_path)
        with open(labels_path) as f:
            self.labels = json.load(f)
        self.history = deque(maxlen=smoothing_window)

    def predict(self, features):
        if not self.available:
            return {'label': 'unknown', 'confidence': 0.0, 'smoothed': 'unknown'}

        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(x)[0]
            idx = int(np.argmax(proba))
            label = self.model.classes_[idx]
            confidence = float(proba[idx])
        else:
            label = self.model.predict(x)[0]
            confidence = 1.0

        self.history.append(label)
        smoothed = max(set(self.history), key=self.history.count)
        return {'label': str(label), 'confidence': confidence, 'smoothed': str(smoothed)}
