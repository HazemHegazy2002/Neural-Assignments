"""
Pipeline 1 - Step 1: Load Images & Extract HOG Features
========================================================
What this step does:
- Loads all 10,000 .bmp images from your dataset folder
- Converts each 28x28 image into a HOG feature vector
- HOG captures the shape/edge structure of each digit
- Saves the features so we can use them in the next steps
"""

import os
import numpy as np
from PIL import Image
from skimage.feature import hog
import time

# ─────────────────────────────────────────────
# CONFIGURATION — change this path to your folder
# ─────────────────────────────────────────────
IMAGE_FOLDER = "Indian_Digits_Train"   # <-- put your folder path here
NUM_IMAGES   = 10000

def load_images(folder):
    """Load all 10,000 images in filename order (1.bmp → 10000.bmp)."""
    print("=" * 50)
    print("Loading images...")
    print("=" * 50)

    images = []
    missing = []

    for i in range(1, NUM_IMAGES + 1):
        path = os.path.join(folder, f"{i}.bmp")

        if not os.path.exists(path):
            missing.append(i)
            continue

        img = Image.open(path).convert("L")        # grayscale
        img_array = np.array(img, dtype=np.float32) / 255.0  # normalize 0-1
        images.append(img_array)

    if missing:
        print(f"WARNING: {len(missing)} images not found: {missing[:5]}...")
    
    print(f"Successfully loaded {len(images)} images")
    print(f"Each image shape: {images[0].shape}")   # should be (28, 28)
    return images


def extract_hog_features(images):
    """
    Convert each 28x28 image into a HOG feature vector.
    
    HOG = Histogram of Oriented Gradients.
    Instead of raw pixels, it captures EDGES and SHAPES.
    
    Why HOG?
    - Raw pixels: 784 numbers, very noisy
    - HOG: captures the structure of the digit (curves, lines)
    - Works much better for clustering similar digits together
    
    HOG settings for 28x28 images:
    - pixels_per_cell = (7, 7)  → divides image into 4x4 grid
    - cells_per_block = (2, 2)  → normalizes contrast locally
    - orientations = 9          → measures 9 gradient directions
    Result: 9 orientations × 4 blocks × 4 = 144 features per image
    """
    print("\n" + "=" * 50)
    print("Extracting HOG features...")
    print("=" * 50)

    start = time.time()
    features = []

    for idx, img in enumerate(images):
        hog_vec = hog(
            img,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
        features.append(hog_vec)

        # Progress update every 1000 images
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} / {NUM_IMAGES} images...")

    features = np.array(features)   # shape: (10000, feature_size)
    elapsed = time.time() - start

    print(f"\nDone! Feature extraction took {elapsed:.1f} seconds")
    print(f"Feature matrix shape: {features.shape}")
    print(f"  → {features.shape[0]} images")
    print(f"  → {features.shape[1]} HOG features per image")

    return features


def save_features(features):
    """Save features to disk so we don't have to recompute them."""
    np.save("hog_features.npy", features)
    print("\nFeatures saved to 'hog_features.npy'")
    print("You can reload them anytime with: np.load('hog_features.npy')")


# ─────────────────────────────────────────────
# MAIN — run this step
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load all images
    images = load_images(IMAGE_FOLDER)

    # 2. Extract HOG features
    features = extract_hog_features(images)

    # 3. Save features for next steps
    save_features(features)

    print("\n" + "=" * 50)
    print("Step 1 COMPLETE")
    print("Next step: K-Means clustering (step2_kmeans.py)")
    print("=" * 50)