# =============================================================
#  Pipeline 2 - Step 2: Data Augmentation
#  Assignment 1 - Part 3
# =============================================================
#
#  Takes the 300 manually labelled seed images and augments each
#  one using:
#    1. Rotation   : +5° and -5°  (2 copies)
#    2. Noise      : Gaussian noise (1 copy)
#    3. Shifts     : up, down, left, right (4 copies)
#  Total per image: 7 augmented copies  →  300 × 7 = 2100 new images
#
#  Outputs:
#    aug_images.npy   — augmented images only  (2100, 784)
#    aug_labels.npy   — their labels           (2100,)
#    train_images.npy — seed + augmented       (2400, 784)
#    train_labels.npy — their labels           (2400,)
#    train_weights.npy— sample weights         (2400,)
#                        seed=100, augmented=1
# =============================================================

import numpy as np
from scipy.ndimage import rotate, shift
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# ── CONFIG ────────────────────────────────────────────────────
IMG_SIZE    = 28
SEED_WEIGHT = 100   # weight for original seed images
AUG_WEIGHT  = 1     # weight for augmented images
NOISE_STD   = 0.05  # std of Gaussian noise
SHIFT_PX    = 2     # pixels to shift in each direction
ROTATION_DEG= 5     # degrees to rotate
# ──────────────────────────────────────────────────────────────


def load_seed_data():
    """Load Step 1 outputs."""
    seed_images  = np.load("seed_images.npy")   # (300, 784)
    seed_labels  = np.load("seed_labels.npy")   # (300,)

    # Only keep fully labelled images (label != -1)
    valid = seed_labels != -1
    seed_images = seed_images[valid]
    seed_labels = seed_labels[valid]

    print(f"Loaded {len(seed_images)} labelled seed images.")
    return seed_images, seed_labels


def augment_single(img_flat, rng):
    """
    Given one flattened image (784,), produce 7 augmented copies.
    Returns list of 7 flat arrays.
    """
    img = img_flat.reshape(IMG_SIZE, IMG_SIZE)
    augmented = []

    # 1. Rotation +5°
    rot_pos = rotate(img, angle=ROTATION_DEG, reshape=False, mode='nearest')
    augmented.append(rot_pos.flatten())

    # 2. Rotation -5°
    rot_neg = rotate(img, angle=-ROTATION_DEG, reshape=False, mode='nearest')
    augmented.append(rot_neg.flatten())

    # 3. Gaussian noise
    noise = img + rng.normal(0, NOISE_STD, img.shape)
    noise = np.clip(noise, 0, 1)
    augmented.append(noise.flatten())

    # 4. Shift up
    sh_up = shift(img, shift=[-SHIFT_PX, 0], mode='nearest')
    augmented.append(sh_up.flatten())

    # 5. Shift down
    sh_dn = shift(img, shift=[SHIFT_PX, 0], mode='nearest')
    augmented.append(sh_dn.flatten())

    # 6. Shift left
    sh_lf = shift(img, shift=[0, -SHIFT_PX], mode='nearest')
    augmented.append(sh_lf.flatten())

    # 7. Shift right
    sh_rt = shift(img, shift=[0, SHIFT_PX], mode='nearest')
    augmented.append(sh_rt.flatten())

    return augmented   # list of 7 arrays, each (784,)


def augment_all(seed_images, seed_labels):
    """Augment all seed images. Returns aug_images, aug_labels."""
    rng = np.random.default_rng(42)
    aug_images = []
    aug_labels = []

    print("Augmenting seed images...")
    for idx, (img, lbl) in enumerate(zip(seed_images, seed_labels)):
        copies = augment_single(img, rng)
        for c in copies:
            aug_images.append(c)
            aug_labels.append(lbl)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(seed_images)} seed images ...")

    aug_images = np.array(aug_images, dtype=np.float32)  # (2100, 784)
    aug_labels = np.array(aug_labels, dtype=int)          # (2100,)

    print(f"  Augmentation done: {len(aug_images)} new images created.")
    return aug_images, aug_labels


def build_training_set(seed_images, seed_labels, aug_images, aug_labels):
    """
    Combine seed + augmented into one training set with weights.
      seed images  → weight = 100
      aug images   → weight = 1
    """
    train_images  = np.vstack([seed_images, aug_images])   # (2400, 784)
    train_labels  = np.concatenate([seed_labels, aug_labels])  # (2400,)
    train_weights = np.concatenate([
        np.full(len(seed_images), SEED_WEIGHT, dtype=float),
        np.full(len(aug_images),  AUG_WEIGHT,  dtype=float)
    ])  # (2400,)

    return train_images, train_labels, train_weights


def save_results(aug_images, aug_labels, train_images, train_labels, train_weights):
    np.save("aug_images.npy",    aug_images)
    np.save("aug_labels.npy",    aug_labels)
    np.save("train_images.npy",  train_images)
    np.save("train_labels.npy",  train_labels)
    np.save("train_weights.npy", train_weights)

    print("\nSaved:")
    for f in ["aug_images.npy", "aug_labels.npy",
              "train_images.npy", "train_labels.npy", "train_weights.npy"]:
        print(f"  {f}")


def preview_augmentations(seed_images, seed_labels, aug_images, aug_labels):
    """
    Show one original image alongside its 7 augmented versions.
    """
    print("\nShowing augmentation preview for first seed image ...")

    original = seed_images[0]
    label    = seed_labels[0]
    # The first 7 rows of aug_images correspond to seed_images[0]
    copies   = aug_images[:7]

    titles = [
        "Original",
        "Rot +5°", "Rot -5°",
        "Noise",
        "Shift Up", "Shift Down",
        "Shift Left", "Shift Right"
    ]
    all_imgs = [original] + list(copies)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle(f"Augmentation Preview  |  Label: {label}", fontsize=13)
    for ax, img, title in zip(axes.flatten(), all_imgs, titles):
        ax.imshow(img.reshape(IMG_SIZE, IMG_SIZE), cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("augmentation_preview.png", dpi=100)
    plt.show(block=True)
    print("  Preview saved as augmentation_preview.png")


def print_summary(seed_images, seed_labels, aug_images, train_images, train_weights):
    print("\n" + "═" * 52)
    print("  STEP 2 SUMMARY")
    print("═" * 52)
    print(f"  Seed images          : {len(seed_images)}")
    print(f"  Augmented copies     : {len(aug_images)}  (7 per seed)")
    print(f"  Total training set   : {len(train_images)}")
    print(f"  Seed weight          : {SEED_WEIGHT}")
    print(f"  Augmented weight     : {AUG_WEIGHT}")
    print(f"  Augmentations used   : Rot ±{ROTATION_DEG}°, Gaussian noise "
          f"(σ={NOISE_STD}), Shift ±{SHIFT_PX}px (4 dirs)")

    print("\n  Label distribution in training set:")
    unique, counts = np.unique(
        np.load("train_labels.npy"), return_counts=True
    )
    for d, c in zip(unique, counts):
        print(f"    Digit {d}: {c:4d}")
    print("═" * 52)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # 1. Load seed data from Step 1
    seed_images, seed_labels = load_seed_data()

    # 2. Augment all seed images
    aug_images, aug_labels = augment_all(seed_images, seed_labels)

    # 3. Build combined training set with weights
    train_images, train_labels, train_weights = build_training_set(
        seed_images, seed_labels, aug_images, aug_labels
    )

    # 4. Save everything
    save_results(aug_images, aug_labels, train_images, train_labels, train_weights)

    # 5. Preview one example
    preview_augmentations(seed_images, seed_labels, aug_images, aug_labels)

    # 6. Summary
    print_summary(seed_images, seed_labels, aug_images, train_images, train_weights)

    print("\nStep 2 done. Run step3_svm_training.py next.")