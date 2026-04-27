"""
utils.py
--------
Helper functions used across the project.

Contains:
    - plot_training_history() : plots loss & accuracy curves from train.py output
    - print_summary_table()   : prints a clean results table for the report
    - save_results_json()     : saves all results to a JSON file for later use
    - set_seed()              : sets random seeds for reproducibility
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch


# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + GPU).
    Call this at the very start of main.py for reproducible runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    print(f"[utils] Random seed set to {seed}")


# ─────────────────────────────────────────────
#  TRAINING CURVES PLOT
# ─────────────────────────────────────────────
def plot_training_history(
    history: dict,
    save_path: str = "training_history.png",
):
    """
    Plot training & validation loss and accuracy curves side by side.

    Parameters
    ----------
    history   : dict returned by train.train()
                Must contain keys: 'train_loss', 'val_loss',
                                   'train_acc',  'val_acc'
    save_path : str — where to save the figure
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History — Arabic Digit CNN", fontsize=14, y=1.01)

    # ── Loss ─────────────────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val Loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", markersize=3, label="Val Acc")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] Training history plot saved to: {save_path}")


# ─────────────────────────────────────────────
#  SUMMARY TABLE  (for report)
# ─────────────────────────────────────────────
def print_summary_table(
    train_history: dict,
    test_results:  dict,
    variation_name: str = "Part (a) — Baseline CNN",
):
    """
    Print a clean summary table suitable for copying into a report.

    Parameters
    ----------
    train_history  : dict returned by train.train()
    test_results   : dict returned by test.evaluate()
    variation_name : label for this experiment
    """
    train_time_ms = train_history.get("total_train_time_ms", 0.0)
    test_time_ms  = test_results.get("test_time_ms", 0.0)
    best_val_acc  = train_history.get("best_val_acc", 0.0)
    test_acc      = test_results.get("test_acc", 0.0)
    per_class     = test_results.get("per_class_acc", [])

    print("\n" + "╔" + "═" * 54 + "╗")
    print(f"║  RESULTS SUMMARY — {variation_name[:33]:<33} ║")
    print("╠" + "═" * 54 + "╣")
    print(f"║  Best Val Accuracy   : {best_val_acc:>6.1f} %                  ║")
    print(f"║  Test  Accuracy      : {test_acc:>6.1f} %                  ║")
    print(f"║  Total Training Time : {train_time_ms:>10.1f} ms              ║")
    print(f"║  Testing  Time       : {test_time_ms:>10.1f} ms              ║")
    print("╠" + "═" * 54 + "╣")
    print("║  Per-Digit Test Accuracy:                            ║")
    for digit, acc in enumerate(per_class):
        bar = "█" * int(acc / 10)
        print(f"║    Digit {digit}: {acc:5.1f}%  {bar:<10}                    ║")
    print("╚" + "═" * 54 + "╝")


# ─────────────────────────────────────────────
#  SAVE RESULTS TO JSON
# ─────────────────────────────────────────────
def save_results_json(
    train_history: dict,
    test_results:  dict,
    variation_name: str = "part_a",
    save_path: str | None = None,
):
    """
    Save train history + test results to a JSON file.
    Useful for comparing multiple variations (parts a, b, c, d).

    Saves to  results_{variation_name}.json  by default.
    """
    if save_path is None:
        save_path = f"results_{variation_name}.json"

    # numpy arrays are not JSON-serialisable — convert to lists
    test_results_clean = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in test_results.items()
    }

    output = {
        "variation":    variation_name,
        "train_history": train_history,
        "test_results":  test_results_clean,
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[utils] Results saved to: {save_path}")


# ─────────────────────────────────────────────
#  COMPARE MULTIPLE VARIATIONS  (for report)
# ─────────────────────────────────────────────
def compare_variations(json_paths: list[str]):
    """
    Load multiple saved result JSON files and print a comparison table.
    Useful when running parts a, b, c, d and comparing side by side.

    Parameters
    ----------
    json_paths : list of paths to results_*.json files
    """
    print("\n" + "─" * 75)
    print(f"{'Variation':<25} {'Test Acc':>10} {'Train Time (ms)':>18} {'Test Time (ms)':>16}")
    print("─" * 75)

    for path in json_paths:
        if not os.path.exists(path):
            print(f"  [!] File not found: {path}")
            continue
        with open(path) as f:
            data = json.load(f)

        name       = data.get("variation", path)
        test_acc   = data["test_results"].get("test_acc", 0.0)
        train_time = data["train_history"].get("total_train_time_ms", 0.0)
        test_time  = data["test_results"].get("test_time_ms", 0.0)

        print(f"  {name:<23} {test_acc:>9.1f}%  {train_time:>16.1f}   {test_time:>14.1f}")

    print("─" * 75)


# ─────────────────────────────────────────────
#  Quick sanity check  (run this file directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(42)

    # Dummy history to test plot
    dummy_history = {
        "train_loss": [1.5, 0.6, 0.3, 0.15, 0.1],
        "val_loss":   [2.7, 0.5, 0.2, 0.10, 0.09],
        "train_acc":  [48,  83,  94,  97,   99],
        "val_acc":    [12,  87,  95,  98,   99],
        "best_val_acc": 99.0,
        "total_train_time_ms": 239761.1,
    }
    dummy_test = {
        "test_acc":      93.3,
        "per_class_acc": [93.3, 96.7, 93.3, 93.3, 86.7, 86.7, 93.3, 90.0, 100.0, 100.0],
        "test_time_ms":  7145.2,
        "confusion_matrix": np.eye(10, dtype=int) * 28,
    }

    plot_training_history(dummy_history, save_path="test_history_plot.png")
    print_summary_table(dummy_history, dummy_test, variation_name="Part (a) — Baseline CNN")
    save_results_json(dummy_history, dummy_test, variation_name="part_a")
    print("\n[utils] All checks passed.")