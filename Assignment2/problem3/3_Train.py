"""
train.py
--------
Training loop for the Arabic digit speech recognition CNN.

Features:
    - Trains DigitCNN using CrossEntropyLoss + Adam optimizer.
    - Evaluates on a validation split (taken from training data) each epoch.
    - ReduceLROnPlateau scheduler: halves LR when val-loss plateaus.
    - Early stopping: stops if val-loss does not improve for PATIENCE epochs.
    - Saves the best model weights to disk (best_model.pth).
    - Returns a history dict for plotting in main.py.
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
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import get_datasets, TRAIN_DIR, TEST_DIR
from model   import get_model


# ─────────────────────────────────────────────
#  HYPERPARAMETERS  –  adjust freely
# ─────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4      # L2 regularisation in Adam
VAL_SPLIT     = 0.15      # fraction of train data used for validation
PATIENCE      = 8         # early-stopping patience (epochs)
SAVE_PATH     = "best_model.pth"


# ─────────────────────────────────────────────
#  TRAINING FUNCTION
# ─────────────────────────────────────────────
def train(
    train_dir: str = TRAIN_DIR,
    test_dir:  str = TEST_DIR,
    # audio / image augmentation hooks (for parts b, c, d)
    audio_augment_kwargs: dict | None = None,
    image_augment_kwargs: dict | None = None,
    save_path: str = SAVE_PATH,
) -> dict:
    """
    Full training run. Returns a history dict with keys:
        'train_loss', 'train_acc', 'val_loss', 'val_acc'  (lists, one value per epoch)
        'best_val_acc'  (float)
        'total_train_time_ms'  (float)
    """

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────
    full_train_ds, _ = get_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        audio_augment_kwargs=audio_augment_kwargs,
        image_augment_kwargs=image_augment_kwargs,
    )

    # Split full train → train + validation
    n_val   = int(len(full_train_ds) * VAL_SPLIT)
    n_train = len(full_train_ds) - n_val
    train_ds, val_ds = random_split(
        full_train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[train] Train samples: {n_train}  |  Val samples: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model     = get_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── History tracking ─────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_loss  = float("inf")
    best_weights   = None
    patience_counter = 0

    # ── Training loop ────────────────────────────────────────────────────
    total_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── TRAIN phase ──────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(specs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * specs.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += specs.size(0)

        train_loss = running_loss / total
        train_acc  = 100.0 * correct / total

        # ── VALIDATION phase ─────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0

        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                logits = model(specs)
                loss   = criterion(logits, labels)

                val_loss_sum += loss.item() * specs.size(0)
                preds         = logits.argmax(dim=1)
                val_correct  += (preds == labels).sum().item()
                val_total    += specs.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = 100.0 * val_correct / val_total

        # ── Record history ───────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── LR scheduler step ────────────────────────────────────────────
        scheduler.step(val_loss)

        # ── Print epoch summary ──────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:6.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:6.2f}%  |  "
            f"LR: {current_lr:.2e}"
        )

        # ── Best model & early stopping ───────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_weights, save_path)
            print(f"  ✓ New best model saved  (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[train] Early stopping triggered after {epoch} epochs "
                      f"(no improvement for {PATIENCE} epochs).")
                break

    total_train_time_ms = (time.perf_counter() - total_start) * 1000
    best_val_acc = max(history["val_acc"])

    print(f"\n[train] Training complete.")
    print(f"[train] Best val accuracy : {best_val_acc:.2f}%")
    print(f"[train] Total training time: {total_train_time_ms:.1f} ms  "
          f"({total_train_time_ms/1000:.1f} s)")
    print(f"[train] Best weights saved to: {save_path}")

    history["best_val_acc"]         = best_val_acc
    history["total_train_time_ms"]  = total_train_time_ms
    return history


# ─────────────────────────────────────────────
#  Run standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train()