"""
step6_part_b_speech.py
----------------------
Part (b) - Speech spectrogram recognition with vs without attention.

Reuses:
  - Spectrogram generation logic from Assignment 2 (1_Dataset.py)
  - DigitCNN and DigitCNNAttention architectures from Assignment 2 (2_model.py)
  - Our own training function (step4_train.py)

Fully self-contained — no fragile external imports.
"""

import os
import re
import json
import time
import copy
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
TRAIN_DIR = r"C:\Neural\Neural-Assignments\Assignment2\data\Train"
TEST_DIR  = r"C:\Neural\Neural-Assignments\Assignment2\data\Test"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    print(f"[info] Seed set to {seed}")

# ─────────────────────────────────────────────
#  SPECTROGRAM CONFIG (same as Assignment 2)
# ─────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 1024
IMAGE_SIZE  = 128
TOP_DB      = 80

# ─────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")

# ─────────────────────────────────────────────
#  LABEL EXTRACTION
# ─────────────────────────────────────────────
def extract_label(filename: str) -> int:
    """
    Extract digit label from filename.
    Supports patterns like: M20n_6.wav → 6, 0_02_41.wav → 0
    """
    stem = os.path.splitext(filename)[0]

    # Try last character before extension
    last_char = stem[-1]
    if last_char.isdigit() and 0 <= int(last_char) <= 9:
        return int(last_char)

    # Try first part split by underscore
    parts = stem.split("_")
    if parts[0].isdigit() and 0 <= int(parts[0]) <= 9:
        return int(parts[0])

    # Regex fallback
    match = re.search(r"([0-9])$", stem)
    if match:
        return int(match.group(1))

    raise ValueError(f"Cannot parse label from: {filename}")

# ─────────────────────────────────────────────
#  WAVEFORM → SPECTROGRAM
# ─────────────────────────────────────────────
def wav_to_spectrogram(filepath: str, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Load WAV file → log-Mel spectrogram → normalize → resize → tensor.
    Identical pipeline to Assignment 2 (1_Dataset.py).

    Returns: float32 tensor of shape [1, image_size, image_size]
    """
    # Load audio
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )

    # Convert to dB and normalize to [0, 1]
    mel_db = librosa.power_to_db(mel, top_db=TOP_DB)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Resize to image_size x image_size
    img = Image.fromarray((mel_db * 255).astype(np.uint8), mode="L")
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # Convert to tensor [1, H, W]
    tensor = torch.tensor(
        np.array(img, dtype=np.float32) / 255.0
    ).unsqueeze(0)

    return tensor

# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class SpeechSpectrogramDataset(Dataset):
    """
    Loads all WAV files from a folder and converts them to
    log-Mel spectrogram tensors. Labels extracted from filenames.
    """
    def __init__(self, folder: str, image_size: int = IMAGE_SIZE):
        self.samples = []
        self.labels  = []
        supported    = (".wav", ".flac", ".mp3", ".ogg")

        files = sorted(os.listdir(folder))
        total = sum(1 for f in files if f.lower().endswith(supported))
        count = 0

        for filename in files:
            if not filename.lower().endswith(supported):
                continue

            count += 1
            if count % 100 == 0:
                print(f"  Loading {count}/{total}...")

            try:
                label    = extract_label(filename)
                filepath = os.path.join(folder, filename)
                spec     = wav_to_spectrogram(filepath, image_size)
                self.samples.append(spec)
                self.labels.append(label)
            except Exception as e:
                print(f"  [warning] Skipping {filename}: {e}")

        from collections import Counter
        dist = Counter(self.labels)
        dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"  Loaded {len(self.samples)} files | {dist_str}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], torch.tensor(self.labels[idx], dtype=torch.long)

# ─────────────────────────────────────────────
#  SPATIAL ATTENTION MODULE
# ─────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (same as Assignment 2 model.py).
    Computes channel-wise avg and max pools, concatenates them,
    and uses a conv layer to produce a spatial attention mask.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels  = 2,
            out_channels = 1,
            kernel_size  = kernel_size,
            padding      = kernel_size // 2,
            bias         = False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled   = torch.cat([avg_pool, max_pool], dim=1)
        attn     = self.sigmoid(self.conv(pooled))
        return x * attn

# ─────────────────────────────────────────────
#  DIGITCNN — NO ATTENTION (Assignment 2)
# ─────────────────────────────────────────────
class DigitCNN(nn.Module):
    """
    VGG-style CNN for spectrogram classification.
    Identical to Assignment 2 (2_model.py) DigitCNN.
    Input:  [B, 1, 128, 128]
    Output: [B, 10] logits
    """
    def __init__(self, num_classes=10, dropout_conv=0.25, dropout_fc=0.5):
        super().__init__()

        self.block1 = self._conv_block(1,   32, dropout_conv)
        self.block2 = self._conv_block(32,  64, dropout_conv)
        self.block3 = self._conv_block(64, 128, dropout_conv)

        # After 3x MaxPool(2x2): 128 → 64 → 32 → 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes)
        )

    @staticmethod
    def _conv_block(in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x):
        x = self.block1(x)   # [B,  32, 64, 64]
        x = self.block2(x)   # [B,  64, 32, 32]
        x = self.block3(x)   # [B, 128, 16, 16]
        x = self.classifier(x)
        return x

# ─────────────────────────────────────────────
#  DIGITCNN — WITH ATTENTION
# ─────────────────────────────────────────────
class DigitCNNWithAttention(nn.Module):
    """
    DigitCNN with spatial attention after each convolutional block.
    Identical to Assignment 2 (2_model.py) DigitCNNAttention.
    Input:  [B, 1, 128, 128]
    Output: [B, 10] logits
    """
    def __init__(self, num_classes=10, dropout_conv=0.25,
                 dropout_fc=0.5, attention_kernel=7):
        super().__init__()

        self.block1 = DigitCNN._conv_block(1,   32, dropout_conv)
        self.block2 = DigitCNN._conv_block(32,  64, dropout_conv)
        self.block3 = DigitCNN._conv_block(64, 128, dropout_conv)

        self.attn1 = SpatialAttention(kernel_size=attention_kernel)
        self.attn2 = SpatialAttention(kernel_size=attention_kernel)
        self.attn3 = SpatialAttention(kernel_size=attention_kernel)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.attn1(x)
        x = self.block2(x)
        x = self.attn2(x)
        x = self.block3(x)
        x = self.attn3(x)
        x = self.classifier(x)
        return x

# ─────────────────────────────────────────────
#  TRAINING FUNCTION
# ─────────────────────────────────────────────
def train_model(model, train_loader, test_loader,
                epochs=50, lr=0.001, weight_decay=1e-4,
                patience=8, description="Model"):
    """
    Full training loop with:
    - CrossEntropyLoss + Adam
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Per-digit accuracy
    - Confusion matrix
    Returns: accuracy, train_time_ms, test_time_ms, train_losses, per_digit_acc, cm
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_losses     = []
    best_val_loss    = float('inf')
    best_weights     = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"TRAINING: {description}")
    print(f"Epochs={epochs}, LR={lr}, WeightDecay={weight_decay}")
    print(f"{'='*60}")

    # ── TRAINING LOOP ──
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

        if (epoch) % 10 == 0:
            print(f"  Epoch [{epoch:3d}/{epochs}] "
                  f"Loss: {avg_loss:.4f}  "
                  f"Train Acc: {train_acc:.1f}%  "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if avg_loss < best_val_loss:
            best_val_loss    = avg_loss
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

    # ── TESTING ──
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

    return accuracy, train_time_ms, test_time_ms, train_losses, per_digit_acc, cm

# ─────────────────────────────────────────────
#  CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def save_confusion_matrix(cm, accuracy, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=range(10), yticklabels=range(10),
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"{title}\nAccuracy: {accuracy:.1f}%", fontsize=13)
    ax.set_xlabel("Predicted Digit", fontsize=11)
    ax.set_ylabel("True Digit",      fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved: {save_path}")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    set_seed(SEED)

    # ── LOAD DATA ──
    print("\n" + "="*60)
    print("LOADING SPEECH DATA")
    print("="*60)
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Test dir : {TEST_DIR}")

    print("\nLoading train data...")
    train_ds = SpeechSpectrogramDataset(TRAIN_DIR)

    print("\nLoading test data...")
    test_ds  = SpeechSpectrogramDataset(TEST_DIR)

    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32,
                              shuffle=False, num_workers=0)

    print(f"\nTrain size : {len(train_ds)}")
    print(f"Test size  : {len(test_ds)}")

    # ── MODEL PARAMETER COUNT ──
    model_base = DigitCNN()
    model_att  = DigitCNNWithAttention()
    params_base = sum(p.numel() for p in model_base.parameters())
    params_att  = sum(p.numel() for p in model_att.parameters())
    print(f"\nDigitCNN parameters              : {params_base:,}")
    print(f"DigitCNNWithAttention parameters : {params_att:,}")

    # ── TRAIN WITHOUT ATTENTION ──
    model_base = DigitCNN()
    (acc_base, tr_base, te_base,
     losses_base, per_digit_base, cm_base) = train_model(
        model_base, train_loader, test_loader,
        epochs=50, lr=0.001, weight_decay=1e-4, patience=8,
        description="DigitCNN — No Attention (Assignment 2 baseline)"
    )

    save_confusion_matrix(
        cm_base, acc_base,
        title="Part (b) — DigitCNN (No Attention)",
        save_path=f"{RESULTS_DIR}/part_b_cm_no_attention.png"
    )

    # ── TRAIN WITH ATTENTION ──
    model_att = DigitCNNWithAttention()
    (acc_att, tr_att, te_att,
     losses_att, per_digit_att, cm_att) = train_model(
        model_att, train_loader, test_loader,
        epochs=50, lr=0.001, weight_decay=1e-4, patience=8,
        description="DigitCNNWithAttention — Spatial Attention"
    )

    save_confusion_matrix(
        cm_att, acc_att,
        title="Part (b) — DigitCNN (With Spatial Attention)",
        save_path=f"{RESULTS_DIR}/part_b_cm_with_attention.png"
    )

    # ── COMPARISON TABLE ──
    print("\n" + "="*70)
    print("PART (b) COMPARISON TABLE — SPEECH SPECTROGRAMS")
    print("="*70)
    print(f"{'Metric':<30} {'No Attention':>18} {'With Attention':>18}")
    print("-"*70)
    print(f"{'Accuracy (%)':<30} {acc_base:>18.1f} {acc_att:>18.1f}")
    print(f"{'Training Time (ms)':<30} {tr_base:>18.1f} {tr_att:>18.1f}")
    print(f"{'Testing Time (ms)':<30} {te_base:>18.1f} {te_att:>18.1f}")
    print(f"{'Parameters':<30} {params_base:>18,} {params_att:>18,}")
    print("-"*70)
    print(f"\n  Per-Digit Accuracy:")
    print(f"  {'Digit':<8} {'No Attention':>14} {'With Attention':>16}")
    print(f"  {'-'*40}")
    for digit in range(10):
        a = per_digit_base.get(str(digit), 0.0)
        b = per_digit_att.get(str(digit), 0.0)
        print(f"  Digit {digit}  {a:>13.1f}%  {b:>14.1f}%")
    print("="*70)

    # ── LOSS CURVES ──
    plt.figure(figsize=(10, 5))
    plt.plot(losses_base, label='No Attention',    color='blue')
    plt.plot(losses_att,  label='With Attention',  color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Part (b) — Speech CNN: Training Loss With vs Without Attention')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/part_b_loss_curves.png", dpi=150)
    plt.close()
    print(f"\n  Loss curves saved: {RESULTS_DIR}/part_b_loss_curves.png")

    # ── SAVE RESULTS TO JSON ──
    results = {
        "part_b_speech": {
            "hyperparameters": {
                "epochs":       50,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "batch_size":   32,
                "patience":     8,
                "optimizer":    "Adam",
                "scheduler":    "ReduceLROnPlateau",
                "image_size":   IMAGE_SIZE,
                "n_mels":       N_MELS,
                "sample_rate":  SAMPLE_RATE
            },
            "without_attention": {
                "accuracy":        round(acc_base, 1),
                "train_time_ms":   round(tr_base,  1),
                "test_time_ms":    round(te_base,  1),
                "parameters":      params_base,
                "per_digit_acc":   per_digit_base,
                "confusion_matrix": cm_base.tolist()
            },
            "with_attention": {
                "accuracy":        round(acc_att, 1),
                "train_time_ms":   round(tr_att,  1),
                "test_time_ms":    round(te_att,  1),
                "parameters":      params_att,
                "per_digit_acc":   per_digit_att,
                "confusion_matrix": cm_att.tolist()
            }
        }
    }

    json_path = f"{RESULTS_DIR}/part_b_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {json_path}")

    print("\n" + "="*60)
    print("✅ Part (b) Complete!")
    print(f"  No Attention  → {acc_base:.1f}%")
    print(f"  With Attention → {acc_att:.1f}%")
    print(f"  Gain/Loss      : {acc_att - acc_base:+.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()