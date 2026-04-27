import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load(csv_path):
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    features = df.drop(columns=['label', 'timestamp']).values.astype(np.float32)
    return features, labels


def build_candidates():
    return {
        'logreg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000)),
        ]),
        'random_forest': Pipeline([
            ('clf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]),
        'mlp': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                  random_state=42)),
        ]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(Path(__file__).parent / 'data' / 'gestures.csv'))
    parser.add_argument('--out-dir', default=str(Path(__file__).parent / 'models'))
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load(args.data)
    print(f'Loaded {len(y)} samples across {len(set(y))} classes')
    print('Per-class counts:')
    for label, count in pd.Series(y).value_counts().items():
        print(f'  {label}: {count}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    results = {}
    best_name, best_model, best_acc = None, None, -1.0

    for name, model in build_candidates().items():
        print(f'\n=== Training {name} ===')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = (preds == y_test).mean()
        print(f'{name} test accuracy: {acc:.4f}')
        print(classification_report(y_test, preds, zero_division=0))
        results[name] = {
            'accuracy': float(acc),
            'report': classification_report(y_test, preds, output_dict=True, zero_division=0),
            'labels': sorted(set(y_test)),
            'confusion_matrix': confusion_matrix(y_test, preds,
                                                 labels=sorted(set(y_test))).tolist(),
        }
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    joblib.dump(best_model, out_dir / 'gesture_model.joblib')
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'best_model': best_name, 'best_accuracy': best_acc,
                   'all_results': results}, f, indent=2)

    labels_sorted = sorted(set(y))
    with open(out_dir / 'labels.json', 'w') as f:
        json.dump(labels_sorted, f, indent=2)

    print(f'\nBest model: {best_name} ({best_acc:.4f})')
    print(f'Saved to {out_dir / "gesture_model.joblib"}')


if __name__ == '__main__':
    main()
