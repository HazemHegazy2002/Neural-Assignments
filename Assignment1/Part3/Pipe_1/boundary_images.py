"""
Pipeline 1 - Step 5: Find Boundary Images
==========================================
What this step does:
- Loads the margins computed in Step 4
- Finds the 30 images the SVM is LEAST confident about
- Shows them to the human one by one
- Human assigns the correct label (weight = 100)
- Saves the updated labels for Step 6 retraining

Why weight = 100?
- Cluster labels from Step 3 have weight = 1
- Human-verified labels here have weight = 100
- This forces the SVM to strongly respect
  these trusted labels when retraining
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMAGE_FOLDER = r"C:\Neural\Indian_Digits_Train" 
N_BOUNDARY      = 40     # how many boundary images to label (20-40)
HUMAN_WEIGHT    = 500    # trust weight for human-verified labels
CLUSTER_WEIGHT  = 1      # trust weight for cluster labels


def load_data():
    """Load everything saved from Steps 3 and 4."""
    print("=" * 50)
    print("Loading data from previous steps...")
    print("=" * 50)

    for f in ["margins.npy", "image_labels.npy",
              "predicted_labels.npy", "scores.npy"]:
        if not os.path.exists(f):
            print(f"ERROR: '{f}' not found! Run Step 4 first.")
            exit()

    margins          = np.load("margins.npy")
    image_labels     = np.load("image_labels.npy")
    predicted_labels = np.load("predicted_labels.npy")
    scores           = np.load("scores.npy")

    print(f"Margins loaded:           {margins.shape}")
    print(f"Image labels loaded:      {image_labels.shape}")
    print(f"Predicted labels loaded:  {predicted_labels.shape}")

    # Load existing trusted indices if any (from previous iterations)
    trusted_file = "trusted_indices.npy"
    if os.path.exists(trusted_file):
        trusted_indices = list(np.load(trusted_file))
        print(f"Previously trusted images: {len(trusted_indices)}")
    else:
        trusted_indices = []
        print("No previously trusted images (first iteration)")

    return margins, image_labels, predicted_labels, scores, trusted_indices


def find_boundary_images(margins, trusted_indices):
    """
    Find the N_BOUNDARY images with the LOWEST margin.

    Low margin = SVM is confused between two digits
    These are the images most likely to be wrong.

    We skip images already labelled by human (trusted_indices)
    because we don't need to re-label them.
    """
    print("\n" + "=" * 50)
    print(f"Finding {N_BOUNDARY} most confused images...")
    print("=" * 50)

    # Sort all images by margin (lowest first = most confused)
    sorted_by_margin = np.argsort(margins)

    # Skip already trusted images
    boundary_indices = []
    for idx in sorted_by_margin:
        if idx not in trusted_indices:
            boundary_indices.append(idx)
        if len(boundary_indices) == N_BOUNDARY:
            break

    boundary_indices = np.array(boundary_indices)

    print(f"Found {len(boundary_indices)} boundary images")
    print(f"Their margins (confidence scores):")
    for i, idx in enumerate(boundary_indices[:5]):
        print(f"  Image {idx+1:5d}: margin = {margins[idx]:.4f}  "
              f"← SVM predicted digit {predicted_labels[idx]}")
    print(f"  ... and {len(boundary_indices)-5} more")

    return boundary_indices


def load_image(image_index):
    """Load a single image by index (0-based)."""
    path = os.path.join(IMAGE_FOLDER, f"{image_index + 1}.bmp")
    img  = Image.open(path).convert("L")
    return np.array(img)


def show_boundary_image(image_idx, margins, predicted_labels, scores, human_count):
    """
    Show one boundary image to the human.
    Display the SVM's top 3 guesses so human knows what SVM thinks.
    """
    img = load_image(image_idx)

    # Get top 3 digit guesses from SVM scores
    img_scores  = scores[image_idx]
    top3_digits = np.argsort(img_scores)[::-1][:3]
    top3_scores = img_scores[top3_digits]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(
        f"Image {image_idx + 1}  |  Human label #{human_count}\n"
        f"SVM guesses: "
        f"{top3_digits[0]}({top3_scores[0]:.2f}), "
        f"{top3_digits[1]}({top3_scores[1]:.2f}), "
        f"{top3_digits[2]}({top3_scores[2]:.2f})\n"
        f"Margin = {margins[image_idx]:.4f}  ← very low = confused",
        fontsize=9
    )

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


def get_human_label_for_image(image_idx, predicted_labels, human_count, total):
    """Ask human to label one boundary image."""
    print(f"\n[{human_count}/{total}] Image {image_idx + 1}")
    print(f"  SVM predicted: {predicted_labels[image_idx]}")
    print(f"  Enter correct digit (0-9)")
    print(f"  Press ENTER to AGREE with SVM prediction")
    print(f"  Enter Q to quit and save progress")

    while True:
        user_input = input("  Your label: ").strip().upper()

        # Just pressed ENTER → agree with SVM
        if user_input == "":
            label = int(predicted_labels[image_idx])
            print(f"  → Agreed with SVM: {label}")
            return label

        if user_input == 'Q':
            return -2  # quit signal

        try:
            label = int(user_input)
            if 0 <= label <= 9:
                if label == predicted_labels[image_idx]:
                    print(f"  → Confirmed SVM was correct: {label}")
                else:
                    print(f"  → Corrected SVM: {predicted_labels[image_idx]}"
                          f" → {label}")
                return label
            else:
                print("  Invalid! Enter 0-9, ENTER to agree, or Q to quit")
        except ValueError:
            print("  Invalid! Enter 0-9, ENTER to agree, or Q to quit")


def label_boundary_images(boundary_indices, margins,
                           predicted_labels, scores,
                           image_labels, trusted_indices):
    """
    Main loop: show each boundary image and get human label.
    """
    print("\n" + "=" * 50)
    print("BOUNDARY IMAGE LABELLING")
    print("=" * 50)
    print("Instructions:")
    print("  - A window shows the image + SVM's top 3 guesses")
    print("  - Type the CORRECT digit (0-9)")
    print("  - Press ENTER if you AGREE with SVM's prediction")
    print("  - Type Q to quit and save progress")
    print(f"  - Each label costs ~10 seconds of your time")
    print(f"  - Total: {N_BOUNDARY} images ≈ {N_BOUNDARY * 10}s "
          f"= {N_BOUNDARY * 10 / 60:.1f} minutes")
    print("=" * 50)

    # Copy labels so we can update them
    updated_labels  = image_labels.copy()

    # Track new trusted images and weights
    new_trusted     = []
    corrections     = 0
    agreements      = 0

    input("\nPress ENTER to start labelling boundary images...")

    for i, img_idx in enumerate(boundary_indices):
        human_count = i + 1

        # Show the image
        show_boundary_image(img_idx, margins, predicted_labels,
                            scores, human_count)

        # Get human label
        label = get_human_label_for_image(
            img_idx, predicted_labels, human_count, N_BOUNDARY)

        # Close image window
        plt.close()

        # Handle quit
        if label == -2:
            print(f"\nSaving progress at image {human_count}...")
            save_progress(updated_labels, trusted_indices + new_trusted)
            print("Progress saved! Run again to continue.")
            break

        # Update label
        old_label = predicted_labels[img_idx]
        updated_labels[img_idx] = label
        new_trusted.append(img_idx)

        if label != old_label:
            corrections += 1
            print(f"  Correction recorded ✓")
        else:
            agreements += 1

        # Progress
        remaining = N_BOUNDARY - human_count
        print(f"  Progress: {human_count}/{N_BOUNDARY} done | "
              f"{remaining} remaining")

    return updated_labels, new_trusted, corrections, agreements


def save_progress(updated_labels, all_trusted_indices):
    """Save intermediate progress."""
    np.save("image_labels.npy",    updated_labels)
    np.save("trusted_indices.npy", np.array(all_trusted_indices))


def save_results(updated_labels, trusted_indices,
                 new_trusted, corrections, agreements):
    """Save all results for Step 6."""
    all_trusted = trusted_indices + new_trusted

    # Save updated labels
    np.save("image_labels.npy",    updated_labels)
    np.save("trusted_indices.npy", np.array(all_trusted))

    # Save weights for ALL images
    # trusted images → weight 100
    # cluster images → weight 1
    weights = np.ones(10000, dtype=float) * CLUSTER_WEIGHT
    for idx in all_trusted:
        weights[idx] = HUMAN_WEIGHT
    np.save("sample_weights.npy", weights)

    print("\n" + "=" * 50)
    print("Saved files:")
    print("  image_labels.npy    ← updated with human corrections")
    print("  trusted_indices.npy ← indices of all human-labelled images")
    print("  sample_weights.npy  ← weight=100 for trusted, weight=1 for rest")
    print(f"\nSummary:")
    print(f"  Images labelled this round: {len(new_trusted)}")
    print(f"  SVM corrections made:       {corrections}")
    print(f"  SVM agreements:             {agreements}")
    print(f"  Total trusted images:       {len(all_trusted)}")
    print(f"  Manual time this round:     {len(new_trusted) * 10}s "
          f"≈ {len(new_trusted) * 10 / 60:.1f} minutes")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data from Steps 3 and 4
    (margins, image_labels, predicted_labels,
     scores, trusted_indices) = load_data()

    # 2. Find the most confused images
    boundary_indices = find_boundary_images(margins, trusted_indices)

    # 3. Human labels the boundary images
    (updated_labels, new_trusted,
     corrections, agreements) = label_boundary_images(
        boundary_indices, margins, predicted_labels,
        scores, image_labels, trusted_indices)

    # 4. Save everything for Step 6
    save_results(updated_labels, trusted_indices,
                 new_trusted, corrections, agreements)

    print("\n" + "=" * 50)
    print("Step 5 COMPLETE ✓")
    print("Next: Step 6 - Retrain SVM with trusted labels")
    print("=" * 50)