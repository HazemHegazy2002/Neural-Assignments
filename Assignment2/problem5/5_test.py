"""
test.py
-------
Evaluates a trained LeNet-5 model on the fixed test set.

Returns:
    - Test accuracy (%)
    - Testing time in milliseconds
    - Per-digit accuracy
"""

import time
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

import torch
import numpy as np
from torch.utils.data import DataLoader

from p5_dataset import ReducedMNIST, SPLITS_DIR
from p5_model   import get_model

# -------------------------------------------------
#  CONFIG
# -------------------------------------------------
BATCH_SIZE      = 32
TEST_SPLIT_PATH = f"{SPLITS_DIR}/test_200.pt"
NUM_CLASSES     = 10


# -------------------------------------------------
#  EVALUATION FUNCTION
# -------------------------------------------------
def evaluate(model, device, verbose=True):
    """
    Evaluate a trained model on the fixed 2000-sample test set.

    Parameters
    ----------
    model   : trained LeNet5 instance (already on device)
    device  : torch.device
    verbose : bool -- if True, print results

    Returns
    -------
    dict with keys:
        'test_acc'       : float  overall accuracy (%)
        'test_time_ms'   : float  pure inference time (ms)
        'per_digit_acc'  : list   per-digit accuracy [0..9]
    """
    # ── Load fixed test set ───────────────────────────────────────────
    test_ds     = ReducedMNIST(TEST_SPLIT_PATH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0,
                             pin_memory=(device.type == "cuda"))

    # ── Inference ─────────────────────────────────────────────────────
    model.eval()
    all_preds  = []
    all_labels = []

    # Measure ONLY the forward-pass time (not data loading)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    if device.type == "cuda":
        torch.cuda.synchronize()
    test_time_ms = (time.perf_counter() - t_start) * 1000

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ── Overall accuracy ──────────────────────────────────────────────
    test_acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)

    # ── Per-digit accuracy ────────────────────────────────────────────
    per_digit_acc = []
    for d in range(NUM_CLASSES):
        mask = all_labels == d
        acc  = 100.0 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
        per_digit_acc.append(round(float(acc), 1))

    if verbose:
        print(f"[test]  Accuracy : {test_acc:.2f}%   "
              f"Time : {test_time_ms:.1f} ms")
        print(f"[test]  Per-digit: "
              + "  ".join(f"{d}:{per_digit_acc[d]:.0f}%" for d in range(NUM_CLASSES)))

    return {
        "test_acc":      round(float(test_acc), 2),
        "test_time_ms":  round(test_time_ms, 1),
        "per_digit_acc": per_digit_acc,
    }


# -------------------------------------------------
#  Run standalone - smoke test (uses last trained model)
# -------------------------------------------------
if __name__ == "__main__":
    from p5_train import train_model

    print("=" * 55)
    print("  Test smoke test: 350 real, no augmentation")
    print("=" * 55)

    # Train quickly
    result  = train_model(n_real=350, verbose=True)
    model   = result["model"]
    device  = result["device"]

    # Evaluate
    metrics = evaluate(model, device, verbose=True)

    print(f"\nTest Accuracy  : {metrics['test_acc']}%")
    print(f"Testing Time   : {metrics['test_time_ms']} ms")