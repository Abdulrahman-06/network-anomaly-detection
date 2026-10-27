import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .compare import RESULTS_DIR
from .data import load_nsl_kdd, split_features_labels
from .train import DATA_DIR, build_pipeline


def anomaly_roc_auc(pipeline, X, y):
    return roc_auc_score(y, -pipeline.score_samples(X))


def main():
    train = load_nsl_kdd(DATA_DIR / "KDDTrain+.txt")
    test = load_nsl_kdd(DATA_DIR / "KDDTest+.txt")
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)

    gradient_boosting = build_pipeline(HistGradientBoostingClassifier(random_state=42)).fit(X_train, y_train)
    isolation_forest = build_pipeline(IsolationForest(n_estimators=200, random_state=42)).fit(X_train[y_train == 0])

    importances = {}
    for name, pipeline, scoring in [
        ("gradient_boosting", gradient_boosting, "roc_auc"),
        ("isolation_forest", isolation_forest, anomaly_roc_auc),
    ]:
        print(f"Measuring feature importance for {name}...")
        result = permutation_importance(pipeline, X_test, y_test, scoring=scoring, n_repeats=3, random_state=42, n_jobs=-1)
        importances[name] = result.importances_mean

    results = pd.DataFrame(importances, index=X_test.columns).round(4)
    for name in importances:
        print(f"\nTop 10 features for {name}:")
        print(results[name].sort_values(ascending=False).head(10).to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    results.sort_values("gradient_boosting", ascending=False).to_csv(RESULTS_DIR / "feature_importance.csv")
    print(f"\nSaved results to {RESULTS_DIR / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
