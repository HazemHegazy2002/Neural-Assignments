"""
train.py
--------
Training loop for the ReducedMNIST LeNet-5 experiments.

Returns:
    - Trained model weights
    - Total training time in milliseconds
    - Final training accuracy
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from p5_dataset import ReducedMNIST, SPLITS_DIR
from p5_model   import get_model


# -------------------------------------------------
#  HYPERPARAMETERS
# -------------------------------------------------
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4


# -------------------------------------------------
#  TRAINING FUNCTION
# -------------------------------------------------
def train_model(n_real, aug_images=None, aug_labels=None, verbose=True):
    """
    Train LeNet-5 on a ReducedMNIST split, optionally with augmented data.

    Parameters
    ----------
    n_real      : int   -- 350, 750, or 1000 (real samples per digit)
    aug_images  : torch.Tensor or None  -- augmented images (M, 1, 28, 28)
    aug_labels  : torch.Tensor or None  -- augmented labels (M,)
    verbose     : bool  -- if True, print epoch-level progress

    Returns
    -------
    dict with keys:
        'model'            : trained LeNet5 instance
        'train_time_ms'    : float  total training wall time (ms)
        'final_train_acc'  : float  accuracy on training set after last epoch
        'device'           : torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build dataset ─────────────────────────────────────────────────
    split_path = f"{SPLITS_DIR}/train_{n_real}.pt"
    dataset    = ReducedMNIST(split_path,
                               extra_images=aug_images,
                               extra_labels=aug_labels)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0,
                        pin_memory=(device.type == "cuda"))

    total_samples = len(dataset)
    if verbose:
        aug_count = len(aug_images) if aug_images is not None else 0
        print(f"[train] real={n_real*10:>6}  aug={aug_count:>6}  "
              f"total={total_samples:>6}  device={device}")

    # ── Model, loss, optimizer ────────────────────────────────────────
    model     = get_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5   # halve LR every 10 epochs
    )

    # ── Training loop ─────────────────────────────────────────────────
    train_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        correct = 0
        total   = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            acc = 100.0 * correct / total
            lr  = optimizer.param_groups[0]["lr"]
            print(f"  Epoch [{epoch:2d}/{NUM_EPOCHS}]  "
                  f"Train Acc: {acc:6.2f}%  LR: {lr:.2e}")

    train_time_ms = (time.perf_counter() - train_start) * 1000
    final_acc     = 100.0 * correct / total

    if verbose:
        print(f"[train] Done. Train Acc: {final_acc:.2f}%  "
              f"Time: {train_time_ms:.1f} ms")

    return {
        "model":           model,
        "train_time_ms":   round(train_time_ms, 1),
        "final_train_acc": round(final_acc, 2),
        "device":          device,
    }


# -------------------------------------------------
#  Run standalone - quick smoke test
# -------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Train smoke test: 350 real, no augmentation")
    print("=" * 55)
    result = train_model(n_real=350, verbose=True)
    print(f"\nFinal train accuracy : {result['final_train_acc']}%")
    print(f"Training time        : {result['train_time_ms']} ms")