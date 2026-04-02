"""
PIPELINE 2 - STEP 2: Data Augmentation
========================================
- Loads the 300 labelled seed images from Step 1
- Applies augmentations to each image:
    1. Rotation: +5° and -5°  (2 copies)
    2. Gaussian noise         (1 copy)
    3. Spatial shifts: up, down, left, right (4 copies)
- Each original image → 7 extra augmented copies
- Total augmented set: 300 × 7 = 2100 images
- Augmented images get weight = 1, seed images get weight = 100
- Extracts HOG features for all augmented images
- Saves everything to pipeline2_step2.pkl for Step 3
"""

import numpy as np
import pickle
from PIL import Image
from scipy.ndimage import rotate, shift
from skimage.feature import hog

# ─────────────────────────────────────────────
#  LOAD STEP 1 DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("Loading Step 1 data...")
print("=" * 55)

with open('pipeline2_step1.pkl', 'rb') as f:
    data = pickle.load(f)

all_images    = data['all_images']      # (10000, 28, 28)
hog_features  = data['hog_features']   # (10000, HOG_DIM)
seed_indices  = data['seed_indices']   # (300,)
seed_images   = data['seed_images']    # (300, 28, 28)
seed_features = data['seed_features']  # (300, HOG_DIM)
seed_labels   = data['seed_labels']    # (300,)
HOG_DIM       = data['HOG_DIM']
HOG_PARAMS    = data['HOG_PARAMS']
IMG_SIZE      = data['IMG_SIZE']

print(f"  Seed images   : {seed_images.shape}")
print(f"  Seed labels   : {seed_labels.shape}")
print(f"  HOG dimension : {HOG_DIM}\n")

# ─────────────────────────────────────────────
#  AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────────

def aug_rotate(img, angle):
    """Rotate image by given angle (degrees), keep same size."""
    rotated = rotate(img.astype(np.float32), angle, reshape=False, mode='nearest')
    return np.clip(rotated, 0, 255).astype(np.uint8)

def aug_noise(img, std=10.0):
    """Add Gaussian noise."""
    noise = np.random.normal(0, std, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def aug_shift(img, dx=0, dy=0):
    """Shift image by dx pixels horizontally, dy pixels vertically."""
    shifted = shift(img.astype(np.float32), shift=[dy, dx], mode='nearest')
    return np.clip(shifted, 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────
#  AUGMENTATION CONFIG
# ─────────────────────────────────────────────
# Each original → 7 augmented copies
AUGMENTATIONS = [
    ('rotate+5',  lambda img: aug_rotate(img, +5)),
    ('rotate-5',  lambda img: aug_rotate(img, -5)),
    ('noise',     lambda img: aug_noise(img, std=10.0)),
    ('shift_up',  lambda img: aug_shift(img, dy=-2)),
    ('shift_down',lambda img: aug_shift(img, dy=+2)),
    ('shift_left',lambda img: aug_shift(img, dx=-2)),
    ('shift_right',lambda img: aug_shift(img, dx=+2)),
]

N_AUG      = len(AUGMENTATIONS)          # 7
TOTAL_AUG  = len(seed_images) * N_AUG   # 300 × 7 = 2100

print("=" * 55)
print(f"STEP 2: Augmenting {len(seed_images)} seed images...")
print(f"        {N_AUG} augmentations each → {TOTAL_AUG} new images")
print("=" * 55)

# ─────────────────────────────────────────────
#  GENERATE AUGMENTED IMAGES + HOG FEATURES
# ─────────────────────────────────────────────
aug_images   = np.zeros((TOTAL_AUG, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8)
aug_labels   = np.zeros(TOTAL_AUG, dtype=int)

idx = 0
for i, (img, lbl) in enumerate(zip(seed_images, seed_labels)):
    for aug_name, aug_fn in AUGMENTATIONS:
        aug_images[idx] = aug_fn(img)
        aug_labels[idx] = lbl
        idx += 1

    if (i + 1) % 50 == 0:
        print(f"  Augmented {i+1}/{len(seed_images)} seed images...")

print(f"  Augmentation complete. Generated {TOTAL_AUG} images.\n")

# Extract HOG for augmented images
print("Extracting HOG features for augmented images...")
aug_features = np.zeros((TOTAL_AUG, HOG_DIM), dtype=np.float32)

for i in range(TOTAL_AUG):
    aug_features[i] = hog(aug_images[i], **HOG_PARAMS)
    if (i + 1) % 500 == 0:
        print(f"  HOG extracted: {i+1}/{TOTAL_AUG}...")

print(f"  HOG extraction done. Shape: {aug_features.shape}\n")

# ─────────────────────────────────────────────
#  BUILD COMBINED TRAINING SET
# ─────────────────────────────────────────────
# Seed images  → weight 100 (human-verified, trusted)
# Aug images   → weight 1   (synthetic copies)

train_features = np.vstack([seed_features, aug_features])   # (2400, HOG_DIM)
train_labels   = np.concatenate([seed_labels, aug_labels])  # (2400,)
train_weights  = np.concatenate([
    np.full(len(seed_labels), 100, dtype=float),   # seed → weight 100
    np.full(len(aug_labels),    1, dtype=float),   # aug  → weight 1
])

print("=" * 55)
print("Combined Training Set (seed + augmented):")
print("=" * 55)
print(f"  Seed images      : {len(seed_labels):>5}  (weight=100 each)")
print(f"  Augmented images : {len(aug_labels):>5}  (weight=1 each)")
print(f"  Total            : {len(train_labels):>5}")
print(f"  Feature shape    : {train_features.shape}")

# Label distribution
print("\nClass distribution in combined training set:")
for digit in range(10):
    n = int(np.sum(train_labels == digit))
    bar = '█' * (n // 10)
    print(f"  Digit {digit}: {n:4d}  {bar}")

# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────
save_data = {
    # full dataset (needed for later steps)
    'all_images'    : all_images,
    'hog_features'  : hog_features,
    # seed set
    'seed_indices'  : seed_indices,
    'seed_images'   : seed_images,
    'seed_features' : seed_features,
    'seed_labels'   : seed_labels,
    # augmented set
    'aug_images'    : aug_images,
    'aug_features'  : aug_features,
    'aug_labels'    : aug_labels,
    # combined training set for SVM
    'train_features': train_features,
    'train_labels'  : train_labels,
    'train_weights' : train_weights,
    # meta
    'HOG_DIM'       : HOG_DIM,
    'HOG_PARAMS'    : HOG_PARAMS,
    'IMG_SIZE'      : IMG_SIZE,
    'NUM_IMAGES'    : data['NUM_IMAGES'],
    'SEED_SIZE'     : data['SEED_SIZE'],
    # tracking sets for later steps (will grow each iteration)
    'labelled_indices' : list(seed_indices),   # all indices added to training so far
}

with open('pipeline2_step2.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print(f"\nAll data saved → pipeline2_step2.pkl")
print("✓ Ready for Step 3: Initial SVM Training (SVM-1)!")