from pathlib import Path

import joblib
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data import load_nsl_kdd, split_features_labels
from .features import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"


def build_pipeline(model):
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])


def main():
    train = load_nsl_kdd(DATA_DIR / "KDDTrain+.txt")
    test = load_nsl_kdd(DATA_DIR / "KDDTest+.txt")

    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)

    pipeline = build_pipeline(DecisionTreeClassifier(random_state=42))
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions, target_names=["normal", "attack"]))

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "decision_tree.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
