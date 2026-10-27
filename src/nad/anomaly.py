import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPRegressor  # NEW

from .compare import RESULTS_DIR
from .data import load_nsl_kdd, split_features_labels
from .evaluate import recall_on_seen_and_unseen
from .features import build_preprocessor  # NEW
from .train import DATA_DIR, build_pipeline

FALSE_ALARM_RATE = 0.05


def reconstruction_errors(autoencoder, X):  # NEW
    return ((autoencoder.predict(X) - X) ** 2).mean(axis=1)  # NEW


def evaluate_detector(name, normal_scores, attack_scores, y_test, train_classes, test_classes):
    threshold = np.percentile(normal_scores, 100 * (1 - FALSE_ALARM_RATE))
    predictions = (attack_scores > threshold).astype(int)
    return {
        "model": name,
        "test_precision": precision_score(y_test, predictions),
        "test_recall": recall_score(y_test, predictions),
        "test_f1": f1_score(y_test, predictions),
        "test_roc_auc": roc_auc_score(y_test, attack_scores),
        **recall_on_seen_and_unseen(train_classes, test_classes, predictions),
    }


def main():
    train = load_nsl_kdd(DATA_DIR / "KDDTrain+.txt")
    test = load_nsl_kdd(DATA_DIR / "KDDTest+.txt")
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)
    X_normal = X_train[y_train == 0]
    print(f"Training detectors on {len(X_normal)} normal rows, no attacks")

    isolation_forest = build_pipeline(IsolationForest(n_estimators=200, random_state=42))
    isolation_forest.fit(X_normal)

    preprocessor = build_preprocessor().fit(X_normal)  # NEW
    Z_normal = preprocessor.transform(X_normal)  # NEW
    Z_test = preprocessor.transform(X_test)  # NEW
    autoencoder = MLPRegressor(hidden_layer_sizes=(64, 16, 64), max_iter=50, early_stopping=True, random_state=42)  # NEW
    autoencoder.fit(Z_normal, Z_normal)  # NEW

    rows = [
        evaluate_detector(
            "isolation_forest",
            -isolation_forest.score_samples(X_normal),
            -isolation_forest.score_samples(X_test),
            y_test, train["class"], test["class"],
        ),
        evaluate_detector(  # NEW
            "autoencoder",
            reconstruction_errors(autoencoder, Z_normal),
            reconstruction_errors(autoencoder, Z_test),
            y_test, train["class"], test["class"],
        ),
    ]

    results = pd.DataFrame(rows).set_index("model").round(3)
    print(results.to_string())
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "anomaly_detection.csv")
    print(f"Saved results to {RESULTS_DIR / 'anomaly_detection.csv'}")


if __name__ == "__main__":
    main()


