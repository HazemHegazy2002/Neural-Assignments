"""
Pipeline 1 - Step 2: K-Means Clustering
========================================
What this step does:
- Loads the HOG features we saved in Step 1
- Runs K-Means to group all 10,000 images into 50 clusters
- Each cluster should ideally contain images of ONE digit
- Saves the cluster assignments for use in Step 3
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
N_CLUSTERS   = 50    # 10 digits × 5 clusters each
RANDOM_STATE = 42    # makes results reproducible


def load_features():
    """Load the HOG features we saved in Step 1."""
    print("=" * 50)
    print("Loading HOG features from Step 1...")
    print("=" * 50)

    if not os.path.exists("hog_features.npy"):
        print("ERROR: 'hog_features.npy' not found!")
        print("Please run step1_feature_extraction.py first.")
        exit()

    features = np.load("hog_features.npy")
    print(f"Loaded features shape: {features.shape}")
    print(f"  → {features.shape[0]} images")
    print(f"  → {features.shape[1]} HOG features per image")
    return features


def normalize_features(features):
    """
    Normalize features before clustering.

    Why normalize?
    - HOG values can vary a lot between features
    - K-Means uses DISTANCE between points
    - If one feature has values 0-1000 and another 0-1,
      the big one dominates unfairly
    - After normalization: all features have mean=0, std=1
    - This makes K-Means treat all features equally
    """
    print("\n" + "=" * 50)
    print("Normalizing features...")
    print("=" * 50)

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    print(f"Before normalization - mean: {features.mean():.4f}, "
          f"std: {features.std():.4f}")
    print(f"After  normalization - mean: {features_normalized.mean():.4f}, "
          f"std: {features_normalized.std():.4f}")

    return features_normalized, scaler


def run_kmeans(features_normalized):
    """
    Run K-Means clustering.

    How K-Means works:
    1. Place 50 random 'center points' in the feature space
    2. Assign every image to its nearest center
    3. Move each center to the average of its assigned images
    4. Repeat steps 2-3 until centers stop moving

    Result: 50 groups where images inside each group
            are as similar as possible to each other
    """
    print("\n" + "=" * 50)
    print(f"Running K-Means with {N_CLUSTERS} clusters...")
    print("This may take 1-3 minutes, please wait...")
    print("=" * 50)

    start = time.time()

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        init='k-means++',    # smart initialization (better than random)
        n_init=10,           # run 10 times, keep best result
        max_iter=300,        # maximum iterations per run
        random_state=RANDOM_STATE,
        verbose=0
    )

    # This is where the actual clustering happens
    cluster_labels = kmeans.fit_predict(features_normalized)

    elapsed = time.time() - start

    print(f"\nK-Means finished in {elapsed:.1f} seconds")
    print(f"cluster_labels shape: {cluster_labels.shape}")
    print(f"Unique clusters found: {len(np.unique(cluster_labels))}")

    return kmeans, cluster_labels


def analyze_clusters(cluster_labels):
    """
    Show how many images ended up in each cluster.
    Good clusters have roughly equal sizes (~200 images each).
    """
    print("\n" + "=" * 50)
    print("Cluster size analysis:")
    print("=" * 50)

    sizes = []
    for c in range(N_CLUSTERS):
        size = np.sum(cluster_labels == c)
        sizes.append(size)

    sizes = np.array(sizes)
    print(f"Largest  cluster: {sizes.max()} images")
    print(f"Smallest cluster: {sizes.min()} images")
    print(f"Average  cluster: {sizes.mean():.1f} images")
    print(f"(Ideal would be ~{10000 // N_CLUSTERS} images per cluster)")

    print("\nAll cluster sizes:")
    for c in range(N_CLUSTERS):
        bar = "█" * (sizes[c] // 20)   # visual bar
        print(f"  Cluster {c:2d}: {sizes[c]:4d} images  {bar}")

    return sizes


def save_results(kmeans, cluster_labels):
    """Save clustering results for Step 3."""
    np.save("cluster_labels.npy", cluster_labels)
    print("\n" + "=" * 50)
    print("Saved 'cluster_labels.npy'")
    print(f"  → Array of {len(cluster_labels)} numbers")
    print(f"  → Each number (0-{N_CLUSTERS-1}) = which cluster that image belongs to")
    print("\nExample - first 10 images belong to clusters:")
    print(f"  {cluster_labels[:10]}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load HOG features from Step 1
    features = load_features()

    # 2. Normalize features so K-Means works fairly
    features_normalized, scaler = normalize_features(features)

    # 3. Run K-Means clustering
    kmeans, cluster_labels = run_kmeans(features_normalized)

    # 4. Analyze the resulting clusters
    sizes = analyze_clusters(cluster_labels)

    # 5. Save results for Step 3
    save_results(kmeans, cluster_labels)

    print("\n" + "=" * 50)
    print("Step 2 COMPLETE ✓")
    print("Next: Step 3 - Human labelling of clusters")
    print("=" * 50)