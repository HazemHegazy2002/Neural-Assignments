"""
Pipeline 1 - Step 3: Human Labelling of Clusters
==================================================
What this step does:
- For each of the 50 clusters, shows 8 random sample images
- Human looks at them and types which digit (0-9) they are
- That label is assigned to ALL images in the cluster
- Mixed/unclear clusters get label -1 (will be handled by SVM later)
- Saves the assigned labels for all 10,000 images
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMAGE_FOLDER = r"C:\Neural\Indian_Digits_Train"   # <-- your images folder
N_CLUSTERS     = 50
SAMPLES_SHOWN  = 8      # how many images shown per cluster
SECS_PER_CLUSTER = 20   # estimated time per cluster (for reporting)


def load_data():
    """Load cluster labels from Step 2."""
    print("=" * 50)
    print("Loading cluster labels from Step 2...")
    print("=" * 50)

    if not os.path.exists("cluster_labels.npy"):
        print("ERROR: 'cluster_labels.npy' not found!")
        print("Please run step2_kmeans.py first.")
        exit()

    cluster_labels = np.load("cluster_labels.npy")
    print(f"Loaded {len(cluster_labels)} cluster assignments")
    print(f"Clusters range: 0 to {cluster_labels.max()}")
    return cluster_labels


def load_image(image_index):
    """
    Load a single image by its index (0-based).
    image_index 0 → 1.bmp
    image_index 1 → 2.bmp  etc.
    """
    path = os.path.join(IMAGE_FOLDER, f"{image_index + 1}.bmp")
    img = Image.open(path).convert("L")
    return np.array(img)


def show_cluster_samples(cluster_id, image_indices):
    """
    Show 8 random images from a cluster in a matplotlib window.
    The human looks at these and decides the digit label.
    """
    # Pick up to SAMPLES_SHOWN random images from this cluster
    n_show = min(SAMPLES_SHOWN, len(image_indices))
    chosen = np.random.choice(image_indices, size=n_show, replace=False)

    fig, axes = plt.subplots(1, n_show, figsize=(14, 3))
    fig.suptitle(
        f"Cluster {cluster_id}  ({len(image_indices)} images total)\n"
        f"Look at these {n_show} samples — what digit are they?",
        fontsize=13, fontweight='bold'
    )

    for i, img_idx in enumerate(chosen):
        img = load_image(img_idx)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"img {img_idx+1}", fontsize=8)

    plt.tight_layout()
    plt.show(block=False)   # show window without freezing terminal
    plt.pause(0.5)          # small pause so window renders


def get_human_label(cluster_id, n_images):
    """
    Ask the human to type the digit label for this cluster.
    Returns:
        0-9  → valid digit label
        -1   → mixed cluster (skip)
        -2   → quit and save progress
    """
    print(f"\nCluster {cluster_id:2d} | {n_images} images")
    print("  Enter digit (0-9) for this cluster")
    print("  Enter -1 if images are MIXED (different digits)")
    print("  Enter Q  to quit and save progress")

    while True:
        user_input = input("  Your label: ").strip().upper()

        if user_input == 'Q':
            return -2   # quit signal

        try:
            label = int(user_input)
            if label == -1:
                print("  → Marked as MIXED (will be handled by SVM)")
                return -1
            elif 0 <= label <= 9:
                print(f"  → Assigned label: {label}")
                return label
            else:
                print("  Invalid! Please enter 0-9, -1, or Q")
        except ValueError:
            print("  Invalid! Please enter 0-9, -1, or Q")


def label_all_clusters(cluster_labels):
    """
    Main loop: go through all 50 clusters one by one.
    For each cluster:
      1. Show 8 sample images
      2. Human types label
      3. Assign label to all images in cluster
    """
    print("\n" + "=" * 50)
    print("HUMAN LABELLING PHASE")
    print("=" * 50)
    print("Instructions:")
    print("  - A window will open showing 8 images from each cluster")
    print("  - Type the digit (0-9) that best describes the cluster")
    print("  - Type -1 if the cluster contains MIXED digits")
    print("  - Type Q to quit and save progress so far")
    print("  - You can resume later if needed")
    print("=" * 50)

    # Initialize all labels as -1 (unknown) first
    image_labels = np.full(10000, -1, dtype=int)

    # Check if there's saved progress to resume from
    progress_file = "labelling_progress.npy"
    start_cluster = 0

    if os.path.exists(progress_file):
        resume = input("\nFound saved progress. Resume? (y/n): ").strip().lower()
        if resume == 'y':
            saved = np.load(progress_file, allow_pickle=True).item()
            image_labels  = saved['image_labels']
            start_cluster = saved['next_cluster']
            print(f"Resuming from cluster {start_cluster}")

    # Track stats
    manual_time_seconds = 0
    mixed_clusters      = []
    labelled_clusters   = {}

    input("\nPress ENTER to start labelling...")

    for cluster_id in range(start_cluster, N_CLUSTERS):

        # Find all image indices belonging to this cluster
        image_indices = np.where(cluster_labels == cluster_id)[0]

        # Show sample images in a popup window
        show_cluster_samples(cluster_id, image_indices)

        # Get human label
        label = get_human_label(cluster_id, len(image_indices))

        # Close the matplotlib window
        plt.close()

        # Handle quit signal
        if label == -2:
            print(f"\nSaving progress at cluster {cluster_id}...")
            np.save(progress_file, {
                'image_labels': image_labels,
                'next_cluster': cluster_id
            })
            print("Progress saved! Run again to resume.")
            break

        # Assign label to all images in this cluster
        if label == -1:
            mixed_clusters.append(cluster_id)
            # images stay as -1 (unknown)
        else:
            image_labels[image_indices] = label
            labelled_clusters[cluster_id] = label

        # Track time
        manual_time_seconds += SECS_PER_CLUSTER

        # Show progress
        done = cluster_id - start_cluster + 1
        remaining = N_CLUSTERS - cluster_id - 1
        print(f"  Progress: {done}/{N_CLUSTERS - start_cluster} clusters done | "
              f"{remaining} remaining | "
              f"Time so far: {manual_time_seconds}s")

    return image_labels, labelled_clusters, mixed_clusters, manual_time_seconds


def print_summary(image_labels, labelled_clusters, mixed_clusters, manual_time):
    """Print a summary of the labelling results."""
    print("\n" + "=" * 50)
    print("LABELLING SUMMARY")
    print("=" * 50)

    labelled_images = np.sum(image_labels >= 0)
    unknown_images  = np.sum(image_labels == -1)

    print(f"Clusters labelled:    {len(labelled_clusters)} / {N_CLUSTERS}")
    print(f"Mixed clusters:       {len(mixed_clusters)} → {mixed_clusters}")
    print(f"Images with label:    {labelled_images}")
    print(f"Images unknown (-1):  {unknown_images}")
    print(f"Manual time:          {manual_time}s "
          f"≈ {manual_time/60:.1f} minutes")
    print(f"(vs 27.8 hours for fully manual labelling)")

    print("\nLabel distribution:")
    for digit in range(10):
        count = np.sum(image_labels == digit)
        bar   = "█" * (count // 100)
        print(f"  Digit {digit}: {count:5d} images  {bar}")


def save_results(image_labels):
    """Save the labelled results for Step 4."""
    np.save("image_labels.npy", image_labels)
    print("\n" + "=" * 50)
    print("Saved 'image_labels.npy'")
    print(f"  → Array of 10,000 labels")
    print(f"  → Values 0-9 = digit label")
    print(f"  → Value  -1  = unknown (mixed cluster)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load cluster assignments from Step 2
    cluster_labels = load_data()

    # 2. Go through all clusters and get human labels
    image_labels, labelled_clusters, mixed_clusters, manual_time = \
        label_all_clusters(cluster_labels)

    # 3. Print summary
    print_summary(image_labels, labelled_clusters, mixed_clusters, manual_time)

    # 4. Save for Step 4
    save_results(image_labels)

    print("\n" + "=" * 50)
    print("Step 3 COMPLETE ✓")
    print("Next: Step 4 - Train initial RBF-SVM")
    print("=" * 50)