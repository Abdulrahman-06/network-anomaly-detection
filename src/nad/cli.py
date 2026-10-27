import argparse

from . import anomaly, compare, explain, plot, train, tune

COMMANDS = {
    "train": (train.main, "Train the decision tree and save it to models/"),
    "compare": (compare.main, "Compare supervised models with cross-validation"),
    "anomaly": (anomaly.main, "Evaluate Isolation Forest and the autoencoder"),
    "tune": (tune.main, "Tune the cutoff for random forest and gradient boosting"),
    "plot": (plot.main, "Draw the recall vs false alarm curves"),
    "explain": (explain.main, "Measure which features the models rely on"),
}


def main():
    parser = argparse.ArgumentParser(prog="nad", description="Network anomaly detection on the NSL-KDD dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        subparsers.add_parser(name, help=description)
    subparsers.add_parser("all", help="Run every command in order")
    args = parser.parse_args()

    if args.command == "all":
        for name, (run, _) in COMMANDS.items():
            print(f"\n=== nad {name} ===")
            run()
    else:
        run, _ = COMMANDS[args.command]
        run()
