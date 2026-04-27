"""
experiments.py
--------------
Runs all 12 experiments required by Problem 5 and fills the results table.

Experiment grid:
                    350 real    750 real    1000 real
    0 generated  |    (1)         (2)         (3)
 1000 generated  |    (4)         (5)         (6)
 1500 generated  |    (7)         (8)         (9)
 2000 generated  |   (10)        (11)        (12)

For each cell we record:
    - Test accuracy (%)
    - Training time (ms)
    - Testing time  (ms)

All 12 experiments use the SAME fixed test set (test_200.pt).
"""

import os
import json
import time
import torch

from p5_augment     import get_augmented_data
from p5_train       import train_model
from p5_test        import evaluate


# -------------------------------------------------
#  EXPERIMENT GRID  (matches assignment table)
# -------------------------------------------------
REAL_SIZES = [350, 750, 1000]        # columns
AUG_SIZES  = [0, 1000, 1500, 2000]  # rows

RESULTS_PATH = "results/experiment_results.json"


# -------------------------------------------------
#  RUN ONE EXPERIMENT
# -------------------------------------------------
def run_one(n_real, n_aug, verbose=True):
    """
    Run a single experiment: train on (n_real + n_aug) samples per digit,
    evaluate on fixed test set.

    Returns dict with accuracy and timing.
    """
    label = f"real={n_real:4d}  aug={n_aug:4d}"
    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Experiment: {label}")
        print(f"{'─'*55}")

    # Step 1 — generate augmented data (returns None if n_aug == 0)
    aug_images, aug_labels = get_augmented_data(n_real, n_aug)

    # Step 2 — train
    train_result = train_model(
        n_real=n_real,
        aug_images=aug_images,
        aug_labels=aug_labels,
        verbose=verbose,
    )

    # Step 3 — evaluate on fixed test set
    metrics = evaluate(
        model=train_result["model"],
        device=train_result["device"],
        verbose=verbose,
    )

    return {
        "n_real":          n_real,
        "n_aug":           n_aug,
        "test_acc":        metrics["test_acc"],
        "train_time_ms":   train_result["train_time_ms"],
        "test_time_ms":    metrics["test_time_ms"],
        "per_digit_acc":   metrics["per_digit_acc"],
        "final_train_acc": train_result["final_train_acc"],
    }


# -------------------------------------------------
#  RUN ALL 12 EXPERIMENTS
# -------------------------------------------------
def run_all(verbose=True):
    """
    Run all 12 experiments and return a nested dict of results.

    results[n_real][n_aug] = {test_acc, train_time_ms, test_time_ms, ...}
    """
    os.makedirs("results", exist_ok=True)

    results = {}
    total   = len(REAL_SIZES) * len(AUG_SIZES)
    count   = 0

    wall_start = time.perf_counter()

    for n_real in REAL_SIZES:
        results[n_real] = {}
        for n_aug in AUG_SIZES:
            count += 1
            print(f"\n[experiments] Running {count}/{total} ...")
            result = run_one(n_real, n_aug, verbose=verbose)
            results[n_real][n_aug] = result

    wall_time = (time.perf_counter() - wall_start) / 60

    # Save to JSON
    # Convert int keys to str for JSON compatibility
    json_results = {
        str(k): {str(k2): v2 for k2, v2 in v.items()}
        for k, v in results.items()
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n[experiments] All {total} experiments done in {wall_time:.1f} min")
    print(f"[experiments] Results saved to {RESULTS_PATH}")

    return results


# -------------------------------------------------
#  PRINT RESULTS TABLE
# -------------------------------------------------
def print_table(results):
    """
    Print the assignment results table in a clean format.

    Accuracy table:
                  350 real   750 real  1000 real
    0 generated |   xx.x%    xx.x%     xx.x%
 1000 generated |   xx.x%    xx.x%     xx.x%
 1500 generated |   xx.x%    xx.x%     xx.x%
 2000 generated |   xx.x%    xx.x%     xx.x%
    """
    col_w = 12

    # ── Accuracy table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS TABLE — TEST ACCURACY (%)")
    print("=" * 60)
    header = f"{'Generated':>14} |" + "".join(f"{'real='+str(r):>{col_w}}" for r in REAL_SIZES)
    print(header)
    print("-" * (14 + 1 + col_w * len(REAL_SIZES)))
    for n_aug in AUG_SIZES:
        row = f"{str(n_aug)+' gen':>14} |"
        for n_real in REAL_SIZES:
            acc = results[n_real][n_aug]["test_acc"]
            row += f"{acc:>{col_w}.2f}%"
        print(row)

    # ── Training time table ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS TABLE — TRAINING TIME (ms)")
    print("=" * 60)
    print(header)
    print("-" * (14 + 1 + col_w * len(REAL_SIZES)))
    for n_aug in AUG_SIZES:
        row = f"{str(n_aug)+' gen':>14} |"
        for n_real in REAL_SIZES:
            t = results[n_real][n_aug]["train_time_ms"]
            row += f"{t:>{col_w}.0f}"
        print(row)

    # ── Testing time table ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS TABLE — TESTING TIME (ms)")
    print("=" * 60)
    print(header)
    print("-" * (14 + 1 + col_w * len(REAL_SIZES)))
    for n_aug in AUG_SIZES:
        row = f"{str(n_aug)+' gen':>14} |"
        for n_real in REAL_SIZES:
            t = results[n_real][n_aug]["test_time_ms"]
            row += f"{t:>{col_w}.1f}"
        print(row)

    # ── Best result highlight ─────────────────────────────────────────
    print("\n" + "=" * 60)
    best_acc  = 0
    best_conf = None
    for n_real in REAL_SIZES:
        for n_aug in AUG_SIZES:
            acc = results[n_real][n_aug]["test_acc"]
            if acc > best_acc:
                best_acc  = acc
                best_conf = (n_real, n_aug)
    print(f"  BEST: {best_acc:.2f}%  →  "
          f"real={best_conf[0]}/digit  aug={best_conf[1]}/digit")
    print("=" * 60)


# -------------------------------------------------
#  Load and reprint saved results (no retraining)
# -------------------------------------------------
def load_and_print():
    """Load saved JSON results and reprint the table without retraining."""
    if not os.path.exists(RESULTS_PATH):
        print(f"[experiments] No saved results found at {RESULTS_PATH}")
        print("[experiments] Run run_all() first.")
        return

    with open(RESULTS_PATH) as f:
        json_results = json.load(f)

    # Convert str keys back to int
    results = {
        int(k): {int(k2): v2 for k2, v2 in v.items()}
        for k, v in json_results.items()
    }
    print_table(results)
    return results


# -------------------------------------------------
#  Run standalone
# -------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Problem 5 — All Experiments")
    print("=" * 60)

    results = run_all(verbose=True)
    print_table(results)