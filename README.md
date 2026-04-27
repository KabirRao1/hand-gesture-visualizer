# Hand Gesture Visualizer

Real-time hand-gesture recognition that streams MediaPipe landmark data and a trained
gesture classifier to a browser, where Three.js visualizers react to your hands. Pinch
your fingers to drive a corkscrew tunnel, tilt your hand to twist it, or throw a
`thumbs_up` to cycle palettes on a displacement orb.

```
+-----------+     OpenCV     +---------------+    WebSocket    +-----------------+
|  Webcam   |  ───────────▶  |  landmarks.py |  ─── 30 fps ─▶  |  client.html /  |
+-----------+                |   MediaPipe   |   JSON frames   |    orb.html     |
                             | + sklearn clf |                 |    (Three.js)   |
                             +---------------+                 +-----------------+
```

## Features

- **21-point hand tracking** via MediaPipe Hands.
- **Continuous controls** — pinch-index, pinch-middle, and palm rotation are normalized
  to `0.0–1.0` so visualizers can map them directly to animation parameters.
- **Discrete gesture classifier** — Random Forest trained on normalized landmarks,
  predicts one of 10 gestures (`peace`, `thumbs_up`, `thumbs_down`, `fist`, `open_palm`,
  `ok`, `point`, `rock`, `l_sign`, `none`) with ~97% test accuracy on the included data.
- **Temporal smoothing** — predictions are smoothed across the last 5 frames to reduce
  jitter at the boundary between gestures.
- **WebSocket streaming** at ~30 fps so any browser/Node/Python client can subscribe.
- **Two reference clients** — a neon tunnel that swaps modes on a double-pinch, and a
  displacement-orb scene that responds to discrete gestures.
- **Reproducible ML pipeline** — collect samples, train, evaluate, and deploy with three
  short scripts.

## Project structure

```
ComputerVision/
├── landmarks.py          # WebSocket server + camera loop + classifier inference
├── client.html           # Three.js tunnel visualizer (continuous controls)
├── orb.html              # Three.js displacement orb (discrete gesture-driven)
├── requirements.txt
└── ml/
    ├── features.py       # Landmark normalization (translation/scale/rotation invariant)
    ├── classifier.py     # Inference wrapper with temporal smoothing
    ├── collect.py        # Press 1–9/0 to label live frames into a CSV
    ├── train.py          # Trains LogReg / RandomForest / MLP, picks the best
    ├── data/
    │   └── gestures.csv          # Collected feature samples (label + 63 floats)
    └── models/
        ├── gesture_model.joblib  # Trained sklearn pipeline
        ├── labels.json           # Ordered class labels
        └── metrics.json          # Per-model accuracy + classification report
```

## Prerequisites

- Python 3.10+ (the code relies on standard `asyncio` and modern `pathlib` APIs).
- A working webcam.
- A modern browser (Chrome/Edge/Firefox) for the Three.js clients.

> MediaPipe ships native binaries; on Windows make sure you're using a 64-bit Python.

## Installation

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/ComputerVision.git
cd ComputerVision

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Quick start (use the included model)

The repository ships with a trained classifier (`ml/models/gesture_model.joblib`) so you
can run the full pipeline without retraining.

1. **Start the server** — opens your webcam and starts the WebSocket on port `8765`:

   ```bash
   python landmarks.py
   ```

   You should see a window titled "Feed" with green dots on your hand and a HUD showing
   `Pinch Index`, `Pinch Middle`, `Rotation`, and `Gesture (confidence)`. Press `q` in the
   feed window to quit.

2. **Open a visualizer** — double-click [client.html](client.html) (the neon tunnel) or
   [orb.html](orb.html) (the displacement orb). Both connect to `ws://localhost:8765`
   automatically.

   - **Tunnel:** pinch your index to control speed, pinch your middle to shift hue,
     tilt your wrist to twist. Pinch both fingers fully while tilting to swap modes.
   - **Orb:** `thumbs_up` cycles to the next palette, `thumbs_down` cycles back,
     `rock` triggers an impulse.

## Train your own model

If you want to add gestures, collect more data, or tune the classifier:

### 1. Collect samples

```bash
python -m ml.collect
```

A window opens with your webcam feed. The default keybindings are:

| Key | Label         | Key | Label         |
| --- | ------------- | --- | ------------- |
| `1` | `peace`       | `6` | `ok`          |
| `2` | `thumbs_up`   | `7` | `point`       |
| `3` | `thumbs_down` | `8` | `rock`        |
| `4` | `fist`        | `9` | `l_sign`      |
| `5` | `open_palm`   | `0` | `none`        |

Each keypress records the current frame as a labeled sample. Aim for **a few hundred
samples per class** with varied positions, distances, and lighting. Press `q` to quit.
Samples are appended to [ml/data/gestures.csv](ml/data/gestures.csv).

> Tip: pass `--frames-per-sample 30` to record 30 frames in a row from a single keypress
> while you slowly rotate / move your hand — much faster than tapping the key repeatedly.

### 2. Train

```bash
python -m ml.train
```

This trains three pipelines (logistic regression, random forest, MLP), evaluates them on
a held-out 20% test split, and saves the best one to
[ml/models/gesture_model.joblib](ml/models/gesture_model.joblib). Per-model accuracy and
the full sklearn classification report are written to
[ml/models/metrics.json](ml/models/metrics.json).

The shipped model is `random_forest` with **96.6% test accuracy** across 10 classes.

### 3. Run

The server will pick up the new model on its next launch — no code changes needed.

## How it works

### Feature extraction (`ml/features.py`)

Each frame, MediaPipe gives 21 `(x, y, z)` landmarks. We normalize them to be invariant
to where the hand is and how it's oriented:

1. **Translate** so the wrist (landmark 0) is at the origin.
2. **Scale** by the wrist→middle-knuckle distance so hand size doesn't matter.
3. **Rotate** the points so the middle-knuckle vector points "up" — this removes
   in-plane rotation, so a `peace` sign is the same feature vector regardless of how
   you tilt your wrist.

The result is a flat 63-dimensional vector per frame.

### Continuous controls (`landmarks.py`)

The HUD-displayed `pinch_index`, `pinch_middle`, and `rotation` are computed analytically
(not from the classifier). They're fingertip-to-thumb distances normalized by hand size,
clamped to `[0, 1]`. The browser smooths them with a LERP for buttery-feeling motion.

### Inference + smoothing (`ml/classifier.py`)

`GestureClassifier.predict()` returns `{label, confidence, smoothed}`. `smoothed` is the
mode of the last 5 predictions — clients should use this for any state machine; raw
`label` is for plotting / debugging.

## WebSocket protocol

The server emits a JSON object per frame on `ws://localhost:8765`:

```json
{
  "pinch_index": 0.42,
  "pinch_middle": 0.18,
  "rotation": 0.51,
  "gesture": "peace",
  "gesture_confidence": 0.93
}
```

All values except `gesture` are `0.0–1.0` floats. `gesture` is one of the strings in
[ml/models/labels.json](ml/models/labels.json), or `"unknown"` if no hand is visible /
the model isn't loaded.

## Building your own visualizer

A minimal client is ~10 lines:

```html
<script>
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (e) => {
  const d = JSON.parse(e.data);
  console.log(d.gesture, d.pinch_index);
  // drive your scene here
};
</script>
```

Both [client.html](client.html) and [orb.html](orb.html) are self-contained references —
copy one and remix the Three.js setup.

## Troubleshooting

| Symptom                                       | Fix                                                                                   |
| --------------------------------------------- | ------------------------------------------------------------------------------------- |
| `cv2.VideoCapture(0)` returns black frames    | Another app is holding the camera. Close Zoom/Teams/Discord and retry.                |
| `No classifier found` warning at startup      | Train one: `python -m ml.collect` then `python -m ml.train`.                          |
| Browser shows "WebSocket connection failed"   | Run `landmarks.py` first; clients refuse to retry once the page loads.                |
| MediaPipe install fails on Python 3.13        | Use Python 3.10–3.12; MediaPipe wheels lag the latest Python release.                 |
| Predictions are jittery                       | Add more training samples for the gesture, especially "edge" poses near other classes.|

## Customization ideas

- **Add a gesture**: extend `GESTURE_KEYS` in [ml/collect.py](ml/collect.py), collect
  samples, retrain. No other code changes required.
- **Two hands**: bump `max_num_hands` in [landmarks.py:13](landmarks.py#L13) and emit a
  list rather than a single object.
- **Lower latency**: drop `await asyncio.sleep(0.033)` in the WebSocket handler for
  unthrottled streaming, or move the camera loop into the asyncio event loop with
  `loop.run_in_executor` to avoid the daemon-thread global state.
- **Different model**: add an entry to `build_candidates()` in [ml/train.py](ml/train.py)
  — anything that exposes `fit/predict_proba` works.

## License

MIT — feel free to use, modify, and share.
