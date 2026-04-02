"""
PIPELINE 2 - STEP 1: Load Dataset, Extract HOG Features & Interactive Seed Labelling
======================================================================================
- Loads all 10,000 images from Indian_Digits_Train folder
- Extracts HOG features from each image
- Randomly samples 300 images as the seed set
- Shows each seed image ONE BY ONE — you type the label (0-9) and press Enter
- Supports: go back (b), save & resume later (q)
- Saves all data + your labels to disk for Step 2
"""

import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')   # works on Windows — change to 'Qt5Agg' if TkAgg fails
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = r"D:\My life\EECE\4th year\2nd term\Neural Networks\Project\Indian_Digits_Train"
SAVE_FILE   = "pipeline2_step1.pkl"
NUM_IMAGES  = 10000
SEED_SIZE   = 300
IMG_SIZE    = (28, 28)
RANDOM_SEED = 42

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    transform_sqrt=True
)

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
#  STEP 1A: Load All Images
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1A: Loading images...")
print("=" * 55)

all_images = np.zeros((NUM_IMAGES, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8)

for i in range(1, NUM_IMAGES + 1):
    img_path = os.path.join(DATA_PATH, f"{i}.bmp")
    img = Image.open(img_path).convert('L')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    all_images[i - 1] = np.array(img)
    if i % 1000 == 0:
        print(f"  Loaded {i}/{NUM_IMAGES} images...")

print(f"Done. Shape: {all_images.shape}\n")

# ─────────────────────────────────────────────
#  STEP 1B: Extract HOG Features
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1B: Extracting HOG features...")
print("=" * 55)

sample_hog = hog(all_images[0], **HOG_PARAMS)
HOG_DIM = len(sample_hog)
print(f"  HOG feature dimension: {HOG_DIM}")

hog_features = np.zeros((NUM_IMAGES, HOG_DIM), dtype=np.float32)
for i in range(NUM_IMAGES):
    hog_features[i] = hog(all_images[i], **HOG_PARAMS)
    if (i + 1) % 1000 == 0:
        print(f"  Extracted {i+1}/{NUM_IMAGES}...")

print(f"Done. Feature matrix: {hog_features.shape}\n")

# ─────────────────────────────────────────────
#  STEP 1C: Random Seed Sampling
# ─────────────────────────────────────────────
seed_indices  = np.sort(np.random.choice(NUM_IMAGES, size=SEED_SIZE, replace=False))
seed_images   = all_images[seed_indices]
seed_features = hog_features[seed_indices]

print(f"Sampled {SEED_SIZE} seed images for manual labelling.\n")

# ─────────────────────────────────────────────
#  STEP 1D: Resume check — load partial labels
# ─────────────────────────────────────────────
seed_labels = np.full(SEED_SIZE, -1, dtype=int)   # -1 = not yet labelled
start_from  = 0

if os.path.exists(SAVE_FILE):
    print("Found existing save file. Checking for partial labels...")
    with open(SAVE_FILE, 'rb') as f:
        saved = pickle.load(f)
    if 'seed_labels' in saved and saved['seed_labels'] is not None:
        prev_labels = saved['seed_labels']
        if len(prev_labels) == SEED_SIZE:
            seed_labels = prev_labels
            labelled    = int(np.sum(seed_labels != -1))
            start_from  = labelled
            print(f"  Resuming from image {start_from + 1} / {SEED_SIZE}\n")
        else:
            print("  Saved labels size mismatch. Starting fresh.\n")
    else:
        print("  No partial labels found. Starting fresh.\n")

# ─────────────────────────────────────────────
#  STEP 1E: Interactive Labelling
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1E: Interactive Labelling")
print("=" * 55)
print("Controls:")
print("  Type 0-9  → assign label, move to next image")
print("  b         → go back to previous image")
print("  q         → save progress and quit (resume later)")
print("=" * 55 + "\n")

fig, ax = plt.subplots(figsize=(4, 4))
plt.ion()
plt.show()

def save_progress():
    data = {
        'all_images'   : all_images,
        'hog_features' : hog_features,
        'seed_indices' : seed_indices,
        'seed_images'  : seed_images,
        'seed_features': seed_features,
        'seed_labels'  : seed_labels,
        'HOG_DIM'      : HOG_DIM,
        'HOG_PARAMS'   : HOG_PARAMS,
        'NUM_IMAGES'   : NUM_IMAGES,
        'SEED_SIZE'    : SEED_SIZE,
        'IMG_SIZE'     : IMG_SIZE,
    }
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(data, f)

i = start_from
while i < SEED_SIZE:

    # ── Display current image ──────────────────
    ax.clear()
    # upscale for better visibility (28x28 is tiny)
    display_img = np.array(Image.fromarray(seed_images[i]).resize((140, 140), Image.NEAREST))
    ax.imshow(display_img, cmap='gray', interpolation='nearest')
    current_lbl = seed_labels[i]
    lbl_str = '?' if current_lbl == -1 else str(current_lbl)
    ax.set_title(
        f"Image {i+1} / {SEED_SIZE}\n"
        f"File: {seed_indices[i]+1}.bmp   |   Current label: {lbl_str}",
        fontsize=11
    )
    ax.axis('off')
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── Get input ──────────────────────────────
    while True:
        try:
            raw = input(f"  [{i+1}/{SEED_SIZE}] Label (0-9 | b=back | q=quit): ").strip().lower()
        except EOFError:
            raw = 'q'

        if raw == 'q':
            save_progress()
            labelled = int(np.sum(seed_labels != -1))
            print(f"\nProgress saved → {SAVE_FILE}")
            print(f"Labelled so far: {labelled} / {SEED_SIZE}")
            print("Run the script again to resume from where you left off.")
            plt.close('all')
            exit()

        elif raw == 'b':
            if i > 0:
                i -= 1
                print(f"  ← Back to image {i+1}")
            else:
                print("  Already at the first image.")
            break

        elif raw in [str(d) for d in range(10)]:
            seed_labels[i] = int(raw)
            # auto-save every 10 labels
            if (i + 1) % 10 == 0:
                save_progress()
                print(f"  ✓ Auto-saved at image {i+1}")
            i += 1
            break

        else:
            print("  ⚠  Invalid input. Please enter 0-9, b, or q.")

plt.close('all')

# ─────────────────────────────────────────────
#  STEP 1F: Labelling Complete — Save Everything
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("✓ All 300 images labelled!")
print("=" * 55)

# Label distribution
unique, counts = np.unique(seed_labels, return_counts=True)
print("\nLabel distribution in seed set:")
for u, c in zip(unique, counts):
    bar = '█' * c
    print(f"  Digit {u}: {c:3d}  {bar}")

save_progress()

print(f"\nAll data saved → {SAVE_FILE}")
print(f"Manual time estimate : {SEED_SIZE * 10} s  ≈  {SEED_SIZE*10/60:.1f} minutes")
print("\n✓ Ready for Step 2: Data Augmentation!")