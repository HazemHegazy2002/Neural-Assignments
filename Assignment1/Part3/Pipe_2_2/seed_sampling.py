# =============================================================
#  Pipeline 2 - Step 1: Random Seed Sampling & Manual Labelling
#  Assignment 1 - Part 3
# =============================================================

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # works reliably on Windows
import matplotlib.pyplot as plt
import time

# ── CONFIG ────────────────────────────────────────────────────
IMAGE_FOLDER   = r"C:\Neural\Indian_Digits_Train"
SEED_SIZE      = 300
SEED_WEIGHT    = 100
TIME_PER_IMAGE = 10
IMG_SIZE       = 28
RANDOM_STATE   = 42
# ──────────────────────────────────────────────────────────────


def load_all_images(folder):
    print("Loading all images from:", folder)
    files = [f for f in os.listdir(folder) if f.lower().endswith('.bmp')]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    images, indices = [], []
    for f in files:
        idx = int(os.path.splitext(f)[0])
        from PIL import Image
        img = Image.open(os.path.join(folder, f)).convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
        indices.append(idx)

    images  = np.array(images)
    indices = np.array(indices)
    print(f"  Loaded {len(images)} images. Shape: {images.shape}")
    return images, indices


def sample_seed(images, indices, seed_size, random_state):
    rng = np.random.default_rng(random_state)
    positions = np.sort(rng.choice(len(images), size=seed_size, replace=False))
    print(f"\nSampled {seed_size} images for manual labelling.")
    return images[positions], indices[positions], positions


def manual_labelling_session(seed_images, seed_indices):
    n      = len(seed_images)
    labels = np.full(n, -1, dtype=int)

    # ── Resume from partial save if exists ──
    if os.path.exists("seed_labels_partial.npy"):
        labels    = np.load("seed_labels_partial.npy")
        start_idx = int(np.sum(labels != -1))
        print(f"\nResuming from image {start_idx + 1}/{n} ...")
    else:
        start_idx = 0
        print(f"\nStarting manual labelling of {n} images.")

    print("─" * 52)
    print("  0-9 = label   |   b = go back   |   q = quit & save")
    print("─" * 52)

    t_start = time.time()
    i = start_idx

    while i < n:
        # ── Show image (non-blocking so terminal stays active) ──
        plt.close('all')
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(seed_images[i].reshape(IMG_SIZE, IMG_SIZE),
                  cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"[{i+1}/{n}]  File: {seed_indices[i]}.bmp", fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        # ── Get input from terminal ──
        while True:
            key = input(f"  [{i+1}/{n}] Label (0-9 | b=back | q=quit): ").strip().lower()

            if key == 'q':
                np.save("seed_labels_partial.npy", labels)
                plt.close('all')
                print(f"\nProgress saved ({int(np.sum(labels!=-1))}/{n} labelled). Re-run to resume.")
                return labels

            elif key == 'b':
                plt.close('all')
                if i > 0:
                    i -= 1
                    print(f"  Back to image {i+1}.")
                else:
                    print("  Already at first image.")
                break

            elif key.isdigit() and 0 <= int(key) <= 9:
                labels[i] = int(key)
                plt.close('all')
                i += 1
                break

            else:
                print("  Invalid — enter a digit 0-9, 'b', or 'q'.")

    elapsed = time.time() - t_start
    print("\n" + "═" * 52)
    print("  Manual labelling COMPLETE!")
    print(f"  Actual time   : {elapsed:.0f} s  ({elapsed/60:.1f} min)")
    print(f"  Expected time : {n * TIME_PER_IMAGE} s  ({n * TIME_PER_IMAGE/60:.1f} min)")
    print("═" * 52)
    return labels


def save_results(seed_images, seed_indices, seed_positions,
                 labels, all_images, all_indices):
    np.save("all_images.npy",     all_images)
    np.save("all_indices.npy",    all_indices)
    np.save("seed_images.npy",    seed_images)
    np.save("seed_indices.npy",   seed_indices)
    np.save("seed_positions.npy", seed_positions)
    np.save("seed_labels.npy",    labels)

    print("\nSaved:")
    for f in ["all_images.npy", "all_indices.npy", "seed_images.npy",
              "seed_indices.npy", "seed_positions.npy", "seed_labels.npy"]:
        print(f"  {f}")


def print_summary(labels):
    labelled = labels != -1
    n_done   = int(np.sum(labelled))
    n_total  = len(labels)

    print("\n" + "═" * 52)
    print("  STEP 1 SUMMARY")
    print("═" * 52)
    print(f"  Seed images      : {n_total}")
    print(f"  Labelled         : {n_done}")
    print(f"  Unlabelled       : {n_total - n_done}")
    print(f"  Seed weight      : {SEED_WEIGHT}  (used in SVM later)")
    print(f"  Manual time est. : {n_done * TIME_PER_IMAGE} s = {n_done * TIME_PER_IMAGE/60:.1f} min")

    if n_done > 0:
        unique, counts = np.unique(labels[labelled], return_counts=True)
        print("\n  Label distribution:")
        for d, c in zip(unique, counts):
            print(f"    Digit {d}: {c:3d}  {'#' * c}")
    print("═" * 52)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # 1. Load all images
    all_images, all_indices = load_all_images(IMAGE_FOLDER)

    # 2. Sample 300 seed images
    seed_images, seed_indices, seed_positions = sample_seed(
        all_images, all_indices, SEED_SIZE, RANDOM_STATE
    )

    # 3. Manual labelling
    seed_labels = manual_labelling_session(seed_images, seed_indices)

    # 4. Save all outputs
    save_results(seed_images, seed_indices, seed_positions,
                 seed_labels, all_images, all_indices)

    # 5. Summary
    print_summary(seed_labels)

    print("\nStep 1 done. Run step2_augmentation.py next.")