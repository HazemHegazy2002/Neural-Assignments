"""
test.py
-------
Evaluates the best saved model on the held-out test set.

Outputs:
    - Overall test accuracy (%)
    - Per-class accuracy for each digit (0–9)
    - Testing time in milliseconds
    - Confusion matrix (printed as table + saved as PNG)
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

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from dataset import get_datasets, TEST_DIR, TRAIN_DIR
from model   import get_model


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BATCH_SIZE  = 32
SAVE_PATH   = "best_model.pth"       # weights saved by train.py
CM_SAVE_PATH = "confusion_matrix.png"
CLASS_NAMES  = [str(i) for i in range(10)]   # ["0", "1", ..., "9"]


# ─────────────────────────────────────────────
#  EVALUATION FUNCTION
# ─────────────────────────────────────────────
def evaluate(
    test_dir:  str = TEST_DIR,
    train_dir: str = TRAIN_DIR,
    weights_path: str = SAVE_PATH,
) -> dict:
    """
    Load best weights and evaluate on the test set.

    Returns
    -------
    dict with keys:
        'test_acc'         : float  — overall accuracy (%)
        'per_class_acc'    : list   — per-digit accuracy (%)
        'test_time_ms'     : float  — total inference time (ms)
        'confusion_matrix' : np.ndarray shape (10, 10)
    """

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] Using device: {device}")

    # ── Load test dataset ────────────────────────────────────────────────
    _, test_ds = get_datasets(train_dir=train_dir, test_dir=test_dir)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0,
                             pin_memory=(device.type == "cuda"))
    print(f"[test] Test samples: {len(test_ds)}")

    # ── Load model + weights ─────────────────────────────────────────────
    model = get_model(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"[test] Loaded weights from: {weights_path}")

    # ── Inference ────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for specs, labels in test_loader:
            specs  = specs.to(device)
            logits = model(specs)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_time_ms = (time.perf_counter() - start_time) * 1000

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Overall accuracy ──────────────────────────────────────────────────
    test_acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)

    # ── Per-class accuracy ────────────────────────────────────────────────
    per_class_acc = []
    for digit in range(10):
        mask     = all_labels == digit
        acc      = 100.0 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
        per_class_acc.append(round(float(acc), 1))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  TEST ACCURACY : {test_acc:.2f}%")
    print(f"  TEST TIME     : {test_time_ms:.1f} ms")
    print("=" * 50)
    print("\n  Per-digit accuracy:")
    for digit, acc in enumerate(per_class_acc):
        bar = "█" * int(acc / 5)
        print(f"    Digit {digit}: {acc:5.1f}%  {bar}")

    print("\n  Confusion Matrix (rows=true, cols=predicted):")
    header = "       " + "  ".join(f"{c:>3}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>3}" for v in row)
        print(f"  [{CLASS_NAMES[i]}]  {row_str}")

    # ── Save confusion matrix as PNG ──────────────────────────────────────
    _save_confusion_matrix(cm, CM_SAVE_PATH)
    print(f"\n[test] Confusion matrix saved to: {CM_SAVE_PATH}")

    return {
        "test_acc":         round(test_acc, 1),
        "per_class_acc":    per_class_acc,
        "test_time_ms":     round(test_time_ms, 1),
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────
#  CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def _save_confusion_matrix(cm: np.ndarray, save_path: str):
    """Save a styled confusion matrix heatmap as a PNG file."""
    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Arabic Digit Recognition\n(rows = true label, cols = predicted)",
                 fontsize=13, pad=15)
    ax.set_xlabel("Predicted Digit", fontsize=11)
    ax.set_ylabel("True Digit",      fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
#  Run standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    results = evaluate()
    print(f"\n[test] Done. Final test accuracy: {results['test_acc']}%")