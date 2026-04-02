"""
Pipeline 3 - Ground Truth Creation
====================================
What this script does:
- Randomly selects 500 images from your 10,000 images
- Shows them to you ONE BY ONE
- You type the correct digit label (0-9)
- Saves progress automatically so you can stop and resume
- Final output: ground_truth_labels.npy and ground_truth_indices.npy

Time required: 500 images x 10 seconds = ~83 minutes
You can split this into multiple sessions!
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMAGE_FOLDER = r"C:\Neural\Indian_Digits_Train"
N_GROUND_TRUTH = 500       # number of images to label
RANDOM_SEED    = 42        # for reproducibility
PROGRESS_FILE  = "ground_truth_progress.npy"
FINAL_FILE_IDX = "ground_truth_indices.npy"
FINAL_FILE_LBL = "ground_truth_labels.npy"


def select_500_images():
    """
    Randomly select 500 image indices from 10,000.
    Uses a fixed random seed so the same 500 images
    are always selected even if you restart.
    """
    np.random.seed(RANDOM_SEED)
    all_indices = np.arange(10000)
    selected    = np.random.choice(all_indices,
                                   size=N_GROUND_TRUTH,
                                   replace=False)
    selected    = np.sort(selected)   # sort for consistency
    return selected


def load_image(image_index):
    """Load image by 0-based index."""
    path = os.path.join(IMAGE_FOLDER, f"{image_index + 1}.bmp")
    img  = Image.open(path).convert("L")
    return np.array(img)


def load_progress():
    """
    Load saved progress if it exists.
    Returns:
        labels      - array of labels assigned so far (-1 = not yet done)
        start_index - which image to resume from
    """
    if os.path.exists(PROGRESS_FILE):
        saved       = np.load(PROGRESS_FILE, allow_pickle=True).item()
        labels      = saved['labels']
        start_index = saved['next_index']
        print(f"Found saved progress!")
        print(f"  Already labelled: {start_index} / {N_GROUND_TRUTH} images")
        print(f"  Resuming from image #{start_index + 1}")
        return labels, start_index
    else:
        labels = np.full(N_GROUND_TRUTH, -1, dtype=int)
        return labels, 0


def save_progress(labels, next_index):
    """Save current progress to disk."""
    np.save(PROGRESS_FILE, {
        'labels':     labels,
        'next_index': next_index
    })


def show_image(image_index, position, total):
    """
    Show a single image in a popup window.
    position = which number we are in the 500 (1-based)
    """
    img = load_image(image_index)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(
        f"Image #{position} of {total}\n"
        f"(file: {image_index + 1}.bmp)\n"
        f"What digit is this? (0-9)",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)


def get_label(position, total, image_index):
    """Ask human to type the correct digit."""
    print(f"\n[{position}/{total}] Image file: {image_index + 1}.bmp")
    print(f"  Type the correct digit (0-9)")
    print(f"  Type Q to save and quit")

    while True:
        user_input = input("  Your label: ").strip().upper()

        if user_input == 'Q':
            return -2   # quit signal

        try:
            label = int(user_input)
            if 0 <= label <= 9:
                print(f"  → Saved label: {label}")
                return label
            else:
                print("  Invalid! Please enter 0-9 or Q")
        except ValueError:
            print("  Invalid! Please enter 0-9 or Q")


def label_500_images(selected_indices, labels, start_index):
    """
    Main labelling loop.
    Shows each image and records human label.
    """
    print("\n" + "=" * 50)
    print("GROUND TRUTH LABELLING")
    print("=" * 50)
    print(f"Total images to label: {N_GROUND_TRUTH}")
    print(f"Starting from:         #{start_index + 1}")
    print(f"Remaining:             {N_GROUND_TRUTH - start_index}")
    print(f"Estimated time:        "
          f"{(N_GROUND_TRUTH - start_index) * 10 / 60:.1f} minutes")
    print("=" * 50)
    print("Tips:")
    print("  - Type Q anytime to save and quit")
    print("  - Run again to resume from where you stopped")
    print("  - Take breaks! Split into multiple sessions")
    print("=" * 50)

    input("\nPress ENTER to start...")

    for i in range(start_index, N_GROUND_TRUTH):

        image_index = selected_indices[i]
        position    = i + 1

        # Show image
        show_image(image_index, position, N_GROUND_TRUTH)

        # Get label
        label = get_label(position, N_GROUND_TRUTH, image_index)

        # Close window
        plt.close()

        # Handle quit
        if label == -2:
            save_progress(labels, i)   # save current position
            print(f"\nProgress saved at image #{i}")
            print(f"Run again to resume from #{i + 1}")
            return labels, False       # False = not finished

        # Save label
        labels[i] = label

        # Auto-save every 10 images
        if position % 10 == 0:
            save_progress(labels, i + 1)
            remaining = N_GROUND_TRUTH - position
            print(f"  Auto-saved. {remaining} images remaining "
                  f"≈ {remaining * 10 / 60:.1f} minutes left")

    return labels, True   # True = finished!


def save_final_results(selected_indices, labels):
    """Save the final ground truth files."""
    np.save(FINAL_FILE_IDX, selected_indices)
    np.save(FINAL_FILE_LBL, labels)

    # Also delete progress file since we're done
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    print("\n" + "=" * 50)
    print("Ground truth saved!")
    print(f"  ground_truth_indices.npy  ← which 500 images")
    print(f"  ground_truth_labels.npy   ← your correct labels")
    print("=" * 50)


def print_summary(selected_indices, labels):
    """Print summary of labelled images."""
    print("\n" + "=" * 50)
    print("LABELLING SUMMARY")
    print("=" * 50)
    print(f"Total images labelled: {np.sum(labels >= 0)} / {N_GROUND_TRUTH}")
    print(f"\nLabel distribution:")
    for digit in range(10):
        count = np.sum(labels == digit)
        bar   = "█" * count
        print(f"  Digit {digit}: {count:3d} images  {bar}")
    print(f"\nExpected ~50 per digit "
          f"(500 images / 10 digits = 50 each)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("Pipeline 3 - Ground Truth Creation")
    print("=" * 50)

    # 1. Select the same 500 images every time (fixed seed)
    selected_indices = select_500_images()
    print(f"Selected 500 images from 10,000")
    print(f"First 5 selected: {selected_indices[:5] + 1}")

    # 2. Load progress if resuming
    labels, start_index = load_progress()

    # 3. Already done?
    if start_index >= N_GROUND_TRUTH:
        print("\nAll 500 images already labelled!")
        print_summary(selected_indices, labels)
    else:
        # 4. Label images
        labels, finished = label_500_images(
            selected_indices, labels, start_index)

        # 5. Save results
        if finished:
            save_final_results(selected_indices, labels)
            print_summary(selected_indices, labels)
            print("\n" + "=" * 50)
            print("GROUND TRUTH COMPLETE ✓")
            print("Next: Use these 500 images to benchmark LLMs")
            print("=" * 50)
        else:
            print("\nResuming later? Just run this script again!")