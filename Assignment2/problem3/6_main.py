"""
main.py
-------
Entry point for Problem 3 — Arabic Digit Speech Recognition.

Runs the full pipeline in order:
    1. Set random seed for reproducibility
    2. Train the CNN on the training set
    3. Evaluate on the held-out test set
    4. Plot training history curves
    5. Print summary table
    6. Save results to JSON (for comparison across parts a, b, c, d)

Usage:
    python main.py

To run with augmentation (parts b / c / d), edit the
audio_augment_kwargs / image_augment_kwargs dicts below.
"""

import os
import sys


def _ensure_project_venv() -> None:
    """Re-run this script with the repo venv Python when available on Windows."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    venv_python = os.path.join(repo_root, ".venv", "Scripts", "python.exe")

    if os.name != "nt":
        return
    if not os.path.isfile(venv_python):
        return
    if os.path.normcase(os.path.abspath(sys.executable)) == os.path.normcase(os.path.abspath(venv_python)):
        return

    script_path = os.path.abspath(__file__)
    os.execv(venv_python, [venv_python, script_path, *sys.argv[1:]])


_ensure_project_venv()

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_local_module(filename: str, module_name: str):
    base_dir = Path(__file__).resolve().parent
    module_path = base_dir / filename
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_utils = _load_local_module("5_utils.py", "utils_impl")
_train = _load_local_module("3_Train.py", "train_impl")
_test = _load_local_module("4_Test.py", "test_impl")

set_seed = _utils.set_seed
plot_training_history = _utils.plot_training_history
print_summary_table = _utils.print_summary_table
save_results_json = _utils.save_results_json

train = _train.train
evaluate = _test.evaluate


# ─────────────────────────────────────────────
#  EXPERIMENT CONFIG  –  edit here per part
# ─────────────────────────────────────────────

# Label used in saved files and summary table
VARIATION_NAME = "part_a"
VARIATION_LABEL = "Part (a) — Baseline CNN (no augmentation)"

# ── Part (a): no augmentation ─────────────────
AUDIO_AUGMENT = None
IMAGE_AUGMENT = None

# ── Part (b): audio augmentation ─────────────
# Uncomment and set VARIATION_NAME = "part_b" to activate
# AUDIO_AUGMENT = {
#     "speed_factor": 1.03,    # speed up 3%
#     "noise_std":    0.005,   # light background noise
# }

# ── Part (c): image augmentation ─────────────
# Uncomment and set VARIATION_NAME = "part_c" to activate
# IMAGE_AUGMENT = {
#     "squeeze_factor":   0.97,   # squeeze horizontally 3%
#     "expand_factor":    1.03,   # expand  horizontally 3%
#     "image_noise_std":  0.02,   # 2% pixel noise
# }

# ── Part (d): both audio + image augmentation ─
# Uncomment both dicts above and set VARIATION_NAME = "part_d"


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  ARABIC DIGIT SPEECH RECOGNITION")
    print(f"  Experiment : {VARIATION_LABEL}")
    print("=" * 60)

    # Step 1 — Reproducibility
    set_seed(42)

    # Step 2 — Train
    print("\n── STEP 1: TRAINING ──────────────────────────────────────")
    history = train(
        audio_augment_kwargs=AUDIO_AUGMENT,
        image_augment_kwargs=IMAGE_AUGMENT,
        save_path=f"best_model_{VARIATION_NAME}.pth",
    )

    # Step 3 — Test
    print("\n── STEP 2: TESTING ───────────────────────────────────────")
    results = evaluate(
        weights_path=f"best_model_{VARIATION_NAME}.pth",
    )

    # Step 4 — Plot training curves
    print("\n── STEP 3: PLOTTING ──────────────────────────────────────")
    plot_training_history(
        history,
        save_path=f"training_history_{VARIATION_NAME}.png",
    )

    # Step 5 — Print summary table
    print("\n── STEP 4: SUMMARY ───────────────────────────────────────")
    print_summary_table(history, results, variation_name=VARIATION_LABEL)

    # Step 6 — Save results JSON
    save_results_json(
        history, results,
        variation_name=VARIATION_NAME,
        save_path=f"results_{VARIATION_NAME}.json",
    )

    print("\n" + "=" * 60)
    print(f"  ALL DONE.")
    print(f"  Test Accuracy  : {results['test_acc']}%")
    print(f"  Training Time  : {history['total_train_time_ms']:.1f} ms")
    print(f"  Testing  Time  : {results['test_time_ms']:.1f} ms")
    print(f"  Saved files    : best_model_{VARIATION_NAME}.pth")
    print(f"                   training_history_{VARIATION_NAME}.png")
    print(f"                   confusion_matrix.png")
    print(f"                   results_{VARIATION_NAME}.json")
    print("=" * 60)


if __name__ == "__main__":
    main()