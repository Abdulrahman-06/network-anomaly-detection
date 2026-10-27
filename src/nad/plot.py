import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPRegressor

from .anomaly import reconstruction_errors
from .compare import RESULTS_DIR
from .data import load_nsl_kdd, split_features_labels
from .features import build_preprocessor
from .train import DATA_DIR, build_pipeline

COLORS = {
    "Random Forest": "#2a78d6",
    "Gradient Boosting": "#eb6834",
    "Isolation Forest": "#1baf7a",
    "Autoencoder": "#eda100",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def attack_scores_by_model(X_train, y_train, X_test):
    X_normal = X_train[y_train == 0]
    scores = {}
    supervised = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    }
    for name, model in supervised.items():
        print(f"Training {name}...")
        pipeline = build_pipeline(model).fit(X_train, y_train)
        scores[name] = pipeline.predict_proba(X_test)[:, 1]

    print("Training Isolation Forest...")
    isolation_forest = build_pipeline(IsolationForest(n_estimators=200, random_state=42)).fit(X_normal)
    scores["Isolation Forest"] = -isolation_forest.score_samples(X_test)

    print("Training Autoencoder...")
    preprocessor = build_preprocessor().fit(X_normal)
    Z_normal = preprocessor.transform(X_normal)
    autoencoder = MLPRegressor(hidden_layer_sizes=(64, 16, 64), max_iter=50, early_stopping=True, random_state=42)
    autoencoder.fit(Z_normal, Z_normal)
    scores["Autoencoder"] = reconstruction_errors(autoencoder, preprocessor.transform(X_test))
    return scores


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("False alarm rate (% of normal traffic flagged)", color=INK_SECONDARY)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)


def main():
    train = load_nsl_kdd(DATA_DIR / "KDDTrain+.txt")
    test = load_nsl_kdd(DATA_DIR / "KDDTest+.txt")
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)
    scores = attack_scores_by_model(X_train, y_train, X_test)

    unseen_or_normal = (test["class"] == "normal") | ~test["class"].isin(train["class"].unique())
    panels = [
        ("All attacks", y_test == y_test),
        ("Attack types never seen in training", unseen_or_normal),
    ]

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, facecolor=SURFACE)
    for ax, (title, rows) in zip(axes, panels):
        style_axes(ax, title)
        for name, model_scores in scores.items():
            false_alarms, recall, _ = roc_curve(y_test[rows], model_scores[rows])
            ax.plot(false_alarms * 100, recall * 100, color=COLORS[name], linewidth=2, label=name)
    axes[0].set_ylabel("Recall (% of attacks caught)", color=INK_SECONDARY)
    axes[0].legend(loc="lower right", frameon=False, labelcolor=INK)
    fig.suptitle("Which model is best depends on how many false alarms you can afford",
                 x=0.06, ha="left", color=INK, fontsize=14)
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "tradeoff_curve.png"
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
