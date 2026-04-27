"""
main.py
-------
Single entry point for Problem 5 — ReducedMNIST Data Augmentation.

Usage:
    python main.py              # run full pipeline (all 12 experiments)
    python main.py --results    # just reprint saved results table
    python main.py --prep       # only prepare data splits, no training
"""

import sys
from p5_dataset     import prepare_splits
from p5_experiments import run_all, print_table, load_and_print


def main():
    # ── Parse simple CLI arguments ────────────────────────────────────
    args = sys.argv[1:]

    if "--results" in args:
        # Just reprint saved results without retraining
        print("=" * 60)
        print("  Problem 5 — Saved Results")
        print("=" * 60)
        load_and_print()
        return

    if "--prep" in args:
        # Only prepare data splits
        print("=" * 60)
        print("  Problem 5 — Data Preparation Only")
        print("=" * 60)
        prepare_splits(force=False)
        print("\n[main] Data ready. Run 'python main.py' to train.")
        return

    # ── Full pipeline ─────────────────────────────────────────────────
    print("=" * 60)
    print("  Problem 5 — ReducedMNIST Data Augmentation")
    print("  Full Pipeline: Prepare → Train → Evaluate")
    print("=" * 60)

    # Step 1 — Make sure data splits exist
    print("\n── STEP 1: DATA PREPARATION ──────────────────────────────")
    prepare_splits(force=False)

    # Step 2 — Run all 12 experiments
    print("\n── STEP 2: RUNNING ALL 12 EXPERIMENTS ────────────────────")
    results = run_all(verbose=True)

    # Step 3 — Print final results table
    print("\n── STEP 3: FINAL RESULTS ──────────────────────────────────")
    print_table(results)

    print("\n[main] All done.")
    print("[main] Reprint table anytime with: python main.py --results")


if __name__ == "__main__":
    main()