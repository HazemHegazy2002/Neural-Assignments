import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_model(model, train_loader, test_loader,
                epochs=20, lr=0.001, weight_decay=1e-4,
                patience=8, description="Model"):
    """
    Full training loop with:
    - CrossEntropyLoss + Adam optimizer
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Per-digit accuracy
    - Confusion matrix
    Returns: accuracy, train_time_ms, test_time_ms,
             train_losses, per_digit_acc, cm
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_losses     = []
    best_loss        = float('inf')
    best_weights     = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"TRAINING: {description}")
    print(f"Epochs={epochs} | LR={lr} | WeightDecay={weight_decay} | Patience={patience}")
    print(f"{'='*60}")

    # ── TRAINING LOOP ──────────────────────────────────────────
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        avg_loss  = epoch_loss / total
        train_acc = 100.0 * correct / total
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 5 == 0:
            print(f"  Epoch [{epoch:3d}/{epochs}] "
                  f"Loss: {avg_loss:.4f}  "
                  f"Train Acc: {train_acc:.1f}%  "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss        = avg_loss
            best_weights     = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    train_time_ms = (time.perf_counter() - train_start) * 1000
    print(f"\n  Training time: {train_time_ms:.1f} ms")

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # ── TESTING ────────────────────────────────────────────────
    test_start = time.perf_counter()
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_time_ms = (time.perf_counter() - test_start) * 1000
    accuracy     = accuracy_score(all_labels, all_preds) * 100

    # Per-digit accuracy
    per_digit_acc = {}
    for digit in range(10):
        mask = np.array(all_labels) == digit
        if mask.sum() > 0:
            digit_acc = accuracy_score(
                np.array(all_labels)[mask],
                np.array(all_preds)[mask]
            ) * 100
            per_digit_acc[str(digit)] = round(digit_acc, 1)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))

    # Print results
    print(f"  Testing time : {test_time_ms:.1f} ms")
    print(f"  Accuracy     : {accuracy:.1f}%")
    print(f"\n  Per-digit accuracy:")
    for digit, acc in per_digit_acc.items():
        bar = "█" * int(acc / 5)
        print(f"    Digit {digit}: {acc:5.1f}%  {bar}")

    return (accuracy, train_time_ms, test_time_ms,
            train_losses, per_digit_acc, cm)


if __name__ == "__main__":
    print("✅ Step 4 Complete — Training function ready!")