"""
Problem 2 - Step 1: Data Loading & Augmentation
================================================
Loads 350 real examples per digit from Reduced MNIST,
applies 10-20x augmentation (rotations, shifts, scaling, noise),
and verifies the dataset before GAN training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Adjust this base path if your Kaggle dataset imports under a different folder name
DATA_ROOT   = "/kaggle/input/datasets/mohamedgamal07/reduced-mnist/Reduced MNIST Data"
TRAIN_DIR   = os.path.join(DATA_ROOT, "Reduced Trainging data")
TEST_DIR    = os.path.join(DATA_ROOT, "Reduced Testing data")
NUM_CLASSES = 10
IMG_SIZE    = 32          # cDCGAN standard (will resize from 28×28)
AUGMENT_FACTOR = 15       # multiply each real image by this many augmented copies
SEED        = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# AUGMENTATION PIPELINE
# ─────────────────────────────────────────────
augment_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),   # shift up to 10% in x and y
        scale=(0.85, 1.15),     # small scaling
    ),
    transforms.ToTensor(),                         # → [0,1]
    transforms.Normalize((0.5,), (0.5,)),          # → [-1,1]  (for GAN)
])

base_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ─────────────────────────────────────────────
# CUSTOM DATASET
# ─────────────────────────────────────────────
class MNISTFolderDataset(Dataset):
    """
    Loads images from  root/<digit>/<image_file>  structure.
    Returns (tensor, label) pairs.
    """
    def __init__(self, root, transform=None, max_per_class=None):
        self.samples   = []
        self.transform = transform

        for digit in range(NUM_CLASSES):
            folder = os.path.join(root, str(digit))
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Folder not found: {folder}")

            files = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ])

            if max_per_class is not None:
                files = files[:max_per_class]

            for fname in files:
                self.samples.append((os.path.join(folder, fname), digit))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")       # force grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────
# AUGMENTED DATASET WRAPPER
# ─────────────────────────────────────────────
class AugmentedDataset(Dataset):
    """
    Wraps a base dataset and returns:
      - the original image (base_transform)
      - AUGMENT_FACTOR augmented copies per original
    effectively multiplying dataset size by (1 + AUGMENT_FACTOR).
    """
    def __init__(self, base_dataset, augment_factor=AUGMENT_FACTOR):
        self.base        = base_dataset          # raw paths + labels
        self.factor      = augment_factor
        self.total       = len(base_dataset) * (1 + augment_factor)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        orig_idx   = idx // (1 + self.factor)
        aug_copy   = idx  % (1 + self.factor)
        path, label = self.base.samples[orig_idx]
        img = Image.open(path).convert("L")

        if aug_copy == 0:
            return base_transform(img), label
        else:
            return augment_transform(img), label


# ─────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────
print("=" * 55)
print("  Step 1 — Data Loading & Augmentation")
print("=" * 55)

# --- Real training set (350 per digit) ---
real_dataset = MNISTFolderDataset(TRAIN_DIR, transform=None, max_per_class=350)
counts_real  = [0] * NUM_CLASSES
for _, lbl in real_dataset.samples:
    counts_real[lbl] += 1

print("\n[Real Training Set]")
print(f"  Total images : {len(real_dataset.samples)}")
for d in range(NUM_CLASSES):
    print(f"  Digit {d}     : {counts_real[d]:4d} images")

# --- Augmented training set ---
aug_dataset = AugmentedDataset(real_dataset, augment_factor=AUGMENT_FACTOR)
print(f"\n[Augmented Training Set]")
print(f"  Augment factor    : {AUGMENT_FACTOR}×  (1 original + {AUGMENT_FACTOR} augmented copies)")
print(f"  Total images      : {len(aug_dataset)}")
print(f"  Per digit (aug)   : {len(aug_dataset) // NUM_CLASSES}")

# --- Test set ---
test_dataset = MNISTFolderDataset(TEST_DIR, transform=base_transform)
counts_test  = [0] * NUM_CLASSES
for _, lbl in test_dataset.samples:
    counts_test[lbl] += 1

print(f"\n[Test Set]")
print(f"  Total images : {len(test_dataset.samples)}")
for d in range(NUM_CLASSES):
    print(f"  Digit {d}     : {counts_test[d]:4d} images")


# ─────────────────────────────────────────────
# VISUALISE REAL vs AUGMENTED SAMPLES
# ─────────────────────────────────────────────
def unnorm(t):
    """Undo [-1,1] normalisation for display."""
    return (t * 0.5 + 0.5).clamp(0, 1)

fig, axes = plt.subplots(6, NUM_CLASSES, figsize=(15, 9))
fig.suptitle("Real (row 1)  vs  Augmented copies (rows 2–6)\n"
             "One column per digit", fontsize=13, fontweight="bold", y=1.01)

for digit in range(NUM_CLASSES):
    # Gather indices for this digit
    digit_indices = [i for i, (_, l) in enumerate(real_dataset.samples) if l == digit]
    base_img_idx  = digit_indices[0]   # first real image of this digit

    # Row 0: original
    orig_img, _ = aug_dataset[base_img_idx * (1 + AUGMENT_FACTOR)]
    axes[0, digit].imshow(unnorm(orig_img).squeeze(), cmap="gray")
    axes[0, digit].set_title(f"Digit {digit}\nReal", fontsize=9)
    axes[0, digit].axis("off")

    # Rows 1-5: augmented copies
    for row in range(1, 6):
        aug_img, _ = aug_dataset[base_img_idx * (1 + AUGMENT_FACTOR) + row]
        axes[row, digit].imshow(unnorm(aug_img).squeeze(), cmap="gray")
        if digit == 0:
            axes[row, digit].set_title(f"Aug {row}", fontsize=7)
        axes[row, digit].axis("off")

plt.tight_layout()
plt.savefig("step1_real_vs_augmented.png", dpi=120, bbox_inches="tight")
plt.show()
print("\n[✓] Saved visualisation → step1_real_vs_augmented.png")


# ─────────────────────────────────────────────
# DATALOADER SMOKE TEST
# ─────────────────────────────────────────────
loader = DataLoader(aug_dataset, batch_size=64, shuffle=True, num_workers=0)
imgs, labels = next(iter(loader))

print(f"\n[DataLoader Smoke Test]")
print(f"  Batch image tensor : {imgs.shape}   (B, C, H, W)")
print(f"  Batch label tensor : {labels.shape}")
print(f"  Pixel range        : [{imgs.min():.3f}, {imgs.max():.3f}]  (expect ≈ −1 to +1)")


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  Real train images   : {len(real_dataset.samples):>6}")
print(f"  Augmented train     : {len(aug_dataset):>6}   ({AUGMENT_FACTOR+1}× real)")
print(f"  Test  images        : {len(test_dataset.samples):>6}")
print(f"  Image size          : {IMG_SIZE}×{IMG_SIZE}")
print(f"  Pixel normalisation : [-1, 1]")
print("=" * 55)
print("\n✅  Step 1 complete — ready for cDCGAN training (Step 2).")