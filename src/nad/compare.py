import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from .data import load_nsl_kdd, split_features_labels
from .evaluate import recall_on_seen_and_unseen
from .train import DATA_DIR, PROJECT_ROOT, build_pipeline

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": HistGradientBoostingClassifier(random_state=42),
}

RESULTS_DIR = PROJECT_ROOT / "results"


def evaluate_on_test(pipeline, X_test, y_test, train_classes, test_classes):
    predictions = pipeline.predict(X_test)
    attack_scores = pipeline.predict_proba(X_test)[:, 1]
    return {
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for name, model in MODELS.items():
        print(f"Evaluating {name}...")
        pipeline = build_pipeline(model)
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=["recall", "f1"], n_jobs=-1)
        pipeline.fit(X_train, y_train)
        rows.append({
            "model": name,
            "cv_recall": cv_scores["test_recall"].mean(),
            "cv_f1": cv_scores["test_f1"].mean(),
            **evaluate_on_test(pipeline, X_test, y_test, train["class"], test["class"]),

        })

    results = pd.DataFrame(rows).set_index("model").round(3)
    print(results.to_string())
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "model_comparison.csv")
    print(f"Saved results to {RESULTS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()