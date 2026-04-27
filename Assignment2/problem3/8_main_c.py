"""
main.py
-------
Entry point for Problem 3 — Arabic Digit Speech Recognition.

CURRENT CONFIG: Part (c) — Image Augmentation
    - Horizontal squeeze 3%  (squeeze_factor = 0.97)
    - Horizontal expand 3%   (expand_factor  = 1.03)
    - Add pixel noise       (image_noise_std = 0.02)

The augmentation is applied only to the TRAINING set.
The TEST set is never augmented.

To switch parts, change VARIATION_NAME, VARIATION_LABEL,
AUDIO_AUGMENT, and IMAGE_AUGMENT below.
"""

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


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


def _load_local_module(filename: str, module_name: str):
    base_dir = Path(__file__).resolve().parent
    module_path = base_dir / filename
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ensure_project_venv()

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
#  EXPERIMENT CONFIG  –  PART (c)
# ─────────────────────────────────────────────
VARIATION_NAME  = "part_c"
VARIATION_LABEL = "Part (c) — Image Augmentation (squeeze/expand ±3% + noise)"

AUDIO_AUGMENT = None

IMAGE_AUGMENT = {
    "squeeze_factor":  0.97,   # squeeze horizontally 3%
    "expand_factor":   1.03,   # expand horizontally 3%
    "image_noise_std": 0.02,   # add 2% noise
}


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

    # Step 4 — Plot
    print("\n── STEP 3: PLOTTING ──────────────────────────────────────")
    plot_training_history(
        history,
        save_path=f"training_history_{VARIATION_NAME}.png",
    )

    # Step 5 — Summary
    print("\n── STEP 4: SUMMARY ───────────────────────────────────────")
    print_summary_table(history, results, variation_name=VARIATION_LABEL)

    # Step 6 — Save
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