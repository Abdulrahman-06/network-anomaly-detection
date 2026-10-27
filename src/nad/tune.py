import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .anomaly import evaluate_detector
from .compare import RESULTS_DIR
from .data import load_nsl_kdd, split_features_labels
from .train import DATA_DIR, build_pipeline

MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": HistGradientBoostingClassifier(random_state=42),
}


def main():
    train = load_nsl_kdd(DATA_DIR / "KDDTrain+.txt")
    test = load_nsl_kdd(DATA_DIR / "KDDTest+.txt")
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for name, model in MODELS.items():
        print(f"Tuning {name}...")
        pipeline = build_pipeline(model)
        out_of_fold = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        pipeline.fit(X_train, y_train)
        rows.append(evaluate_detector(
            f"{name}_tuned",
            out_of_fold[y_train == 0],
            pipeline.predict_proba(X_test)[:, 1],
            y_test, train["class"], test["class"],
        ))

    results = pd.DataFrame(rows).set_index("model").round(3)
    print(results.to_string())
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "threshold_tuning.csv")
    print(f"Saved results to {RESULTS_DIR / 'threshold_tuning.csv'}")


if __name__ == "__main__":
    main()
