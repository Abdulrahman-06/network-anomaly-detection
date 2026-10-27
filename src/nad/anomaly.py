import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .data import load_nsl_kdd, split_features_labels
from .evaluate import recall_on_seen_and_unseen
from .train import DATA_DIR, build_pipeline

FALSE_ALARM_RATE = 0.05

def main():
    train = load_nsl_kdd(DATA_DIR / "KDDTrain+.txt")
    test = load_nsl_kdd(DATA_DIR / "KDDTest+.txt")
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)

    X_normal = X_train[y_train == 0]

    pipeline = build_pipeline(IsolationForest(n_estimators=200, random_state=42))
    pipeline.fit(X_normal)

    normal_scores = -pipeline.score_samples(X_normal)
    threshold = np.percentile(normal_scores, 100 * (1 - FALSE_ALARM_RATE))

    attack_scores = -pipeline.score_samples(X_test)
    predictions = (attack_scores > threshold).astype(int)

    print(f"Trained on {len(X_normal)} normal rows, no attacks")
    print(f"Test ROC-AUC:   {roc_auc_score(y_test, attack_scores):.3f}")
    print(f"Test precision: {precision_score(y_test, predictions):.3f}")
    print(f"Test recall:    {recall_score(y_test, predictions):.3f}")
    print(f"Test F1:        {f1_score(y_test, predictions):.3f}")

    novelty = recall_on_seen_and_unseen(train["class"], test["class"], predictions)
    print(f"Recall on seen attack types:   {novelty['recall_seen']:.3f}")
    print(f"Recall on unseen attack types: {novelty['recall_unseen']:.3f}")



if __name__ == "__main__":
    main()
