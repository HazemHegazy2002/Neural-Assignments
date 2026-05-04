"""
Part (b) - Arabic digit speech recognition with vs without attention.

This script reuses the Assignment 2 spectrogram dataset and training pipeline,
then compares a baseline CNN against a spatial-attention variant.
"""

import os
import sys
from pathlib import Path


def _ensure_project_venv() -> None:
    """Re-run this script with the repo venv Python when available on Windows."""
    repo_root = Path(__file__).resolve().parents[2]
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"

    if os.name != "nt":
        return
    if not venv_python.is_file():
        return
    if os.path.normcase(os.path.abspath(sys.executable)) == os.path.normcase(os.path.abspath(venv_python)):
        return

    script_path = os.path.abspath(__file__)
    os.execv(str(venv_python), [str(venv_python), script_path, *sys.argv[1:]])


def _add_assignment2_problem3_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    problem3_dir = repo_root / "Assignment2" / "problem3"
    if str(problem3_dir) not in sys.path:
        sys.path.insert(0, str(problem3_dir))
    return repo_root


def _print_compare(no_attn: dict, attn: dict) -> None:
    print("\n" + "=" * 70)
    print("PART (b) COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<25} {'No Attention':>20} {'With Attention':>20}")
    print("-" * 70)
    print(f"{'Accuracy (%)':<25} {no_attn['acc']:>20.1f} {attn['acc']:>20.1f}")
    print(f"{'Training Time (ms)':<25} {no_attn['train_ms']:>20.1f} {attn['train_ms']:>20.1f}")
    print(f"{'Testing Time (ms)':<25} {no_attn['test_ms']:>20.1f} {attn['test_ms']:>20.1f}")
    print("=" * 70)


def main() -> None:
    _ensure_project_venv()
    repo_root = _add_assignment2_problem3_to_path()

    train_dir = repo_root / "Assignment2" / "data" / "Train"
    test_dir = repo_root / "Assignment2" / "data" / "Test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Missing data folders. Expected: {train_dir} and {test_dir}"
        )

    from Utils import set_seed
    from train import train
    from test import evaluate

    runs = []
    for model_kind, label in (
        ("baseline", "No Attention"),
        ("attention", "With Attention"),
    ):
        print("\n" + "=" * 60)
        print(f"  ARABIC DIGIT SPEECH RECOGNITION - {label}")
        print("=" * 60)

        set_seed(42)
        weights_path = f"best_model_{model_kind}.pth"

        history = train(
            train_dir=str(train_dir),
            test_dir=str(test_dir),
            model_kind=model_kind,
            save_path=weights_path,
        )

        results = evaluate(
            test_dir=str(test_dir),
            train_dir=str(train_dir),
            model_kind=model_kind,
            weights_path=weights_path,
        )

        runs.append(
            {
                "label": label,
                "acc": results.get("test_acc", 0.0),
                "train_ms": history.get("total_train_time_ms", 0.0),
                "test_ms": results.get("test_time_ms", 0.0),
            }
        )

    _print_compare(runs[0], runs[1])


if __name__ == "__main__":
    main()
