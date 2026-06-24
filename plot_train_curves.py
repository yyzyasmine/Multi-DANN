import argparse
import json
import os

from trainer import _plot_curves


def parse_args():
    parser = argparse.ArgumentParser("Replot train_curves from train_history.json.")
    parser.add_argument("--run_dir", required=True, help="Result directory containing train_history.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.normpath(args.run_dir)
    history_path = os.path.join(run_dir, "train_history.json")
    if not os.path.isfile(history_path):
        raise FileNotFoundError(f"train_history.json not found: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    best_epoch = int(history.get("best_epoch", history["epoch"][-1]))
    paths = _plot_curves(history, run_dir, best_epoch)
    print("[PLOT] train curves saved:")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
