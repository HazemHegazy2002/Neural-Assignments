"""
Pipeline 1 - Step 4: Train Initial RBF-SVM
============================================
What this step does:
- Loads HOG features (Step 1) and image labels (Step 3)
- Filters out unknown images (label = -1)
- Trains an RBF-SVM on the 9,576 labelled images
- Uses the trained SVM to predict labels for ALL 10,000 images
- Computes confidence scores (margin) for every image
- Saves the model predictions and scores for Step 5
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import os
import pickle

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
HUMAN_WEIGHT = 1      # weight for cluster-labelled images
                      # (Step 5 will add trusted images with weight=100)


def load_data():
    """Load HOG features and image labels from previous steps."""
    print("=" * 50)
    print("Loading data from previous steps...")
    print("=" * 50)

    # Check files exist
    for f in ["hog_features.npy", "image_labels.npy"]:
        if not os.path.exists(f):
            print(f"ERROR: '{f}' not found! Run previous steps first.")
            exit()

    features     = np.load("hog_features.npy")       # shape (10000, 324)
    image_labels = np.load("image_labels.npy")        # shape (10000,)

    print(f"Features shape:      {features.shape}")
    print(f"Image labels shape:  {image_labels.shape}")
    print(f"Labelled images:     {np.sum(image_labels >= 0)}")
    print(f"Unknown images (-1): {np.sum(image_labels == -1)}")

    return features, image_labels


def prepare_training_data(features, image_labels):
    """
    Filter out unknown images (label = -1).
    Only use labelled images for training.

    We also normalize the features using StandardScaler:
    - Same normalization as Step 2
    - SVM is sensitive to feature scale
    - All features need to be on the same scale
    """
    print("\n" + "=" * 50)
    print("Preparing training data...")
    print("=" * 50)

    # Get only labelled images (label != -1)
    labelled_mask    = image_labels >= 0
    train_features   = features[labelled_mask]
    train_labels     = image_labels[labelled_mask]
    train_indices    = np.where(labelled_mask)[0]

    print(f"Training samples:  {len(train_features)}")
    print(f"Classes:           {np.unique(train_labels)}")

    # Normalize features
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)

    # Normalize ALL features (for prediction later)
    all_features_norm = scaler.transform(features)

    print(f"Features normalized ✓")

    # Create sample weights (all = 1 for now, Step 5 will add weight=100)
    weights = np.ones(len(train_features), dtype=float) * HUMAN_WEIGHT
    print(f"Sample weights:    all set to {HUMAN_WEIGHT}")

    return (train_features_norm, train_labels,
            train_indices, all_features_norm,
            scaler, weights)


def train_svm(train_features, train_labels, weights):
    """
    Train an RBF-SVM classifier.

    Key parameters:
    - kernel='rbf'    → curved decision boundaries (better for digits)
    - C=10            → how much to penalize mistakes (higher = stricter)
    - gamma='scale'   → how far each training point's influence reaches
    - decision_function_shape='ovr' → one-vs-rest for multiclass

    What 'decision_function_shape' means:
    - We have 10 classes (digits 0-9)
    - SVM trains 10 binary classifiers: "is this a 0 or not?", etc.
    - For each image, each classifier gives a score
    - The class with the highest score wins
    """
    print("\n" + "=" * 50)
    print("Training RBF-SVM...")
    print(f"Training on {len(train_features)} images...")
    print("This may take 2-5 minutes, please wait...")
    print("=" * 50)

    start = time.time()

    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        decision_function_shape='ovr',   # one-vs-rest scores
        random_state=42
    )

    # Train with sample weights
    svm.fit(train_features, train_labels, sample_weight=weights)

    elapsed = time.time() - start
    print(f"\nSVM trained in {elapsed:.1f} seconds ✓")

    return svm


def predict_all(svm, all_features_norm, image_labels):
    """
    Use the trained SVM to predict labels for ALL 10,000 images.
    Also compute confidence scores (margin) for every image.

    What is the margin?
    - SVM gives a score for each of the 10 digit classes
    - scores = [0.1, 0.5, 3.2, 0.4, ...]  ← digit 2 wins (score=3.2)
    - margin = highest score - second highest score
    - margin = 3.2 - 0.5 = 2.7  ← very confident!

    Small margin = SVM is confused between two digits
    Large margin = SVM is very confident about the label
    """
    print("\n" + "=" * 50)
    print("Predicting labels for all 10,000 images...")
    print("=" * 50)

    start = time.time()

    # Get predicted labels
    predicted_labels = svm.predict(all_features_norm)

    # Get decision scores for confidence measurement
    scores = svm.decision_function(all_features_norm)
    # scores shape: (10000, 10) — one score per digit class per image

    elapsed = time.time() - start
    print(f"Prediction done in {elapsed:.1f} seconds ✓")

    # Compute margin = difference between top 2 scores
    # Sort scores descending for each image
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    margins = sorted_scores[:, 0] - sorted_scores[:, 1]
    # margins shape: (10000,) — one margin value per image

    print(f"\nMargin statistics:")
    print(f"  Average margin: {margins.mean():.3f}")
    print(f"  Min margin:     {margins.min():.3f}  ← most confused images")
    print(f"  Max margin:     {margins.max():.3f}  ← most confident images")

    # Show prediction distribution
    print(f"\nPrediction distribution:")
    for digit in range(10):
        count = np.sum(predicted_labels == digit)
        bar   = "█" * (count // 100)
        print(f"  Digit {digit}: {count:5d} images  {bar}")

    return predicted_labels, scores, margins


def check_accuracy_on_labelled(predicted_labels, image_labels):
    """
    Check how well the SVM predicts on the images WE labelled in Step 3.
    This gives us an idea of current performance.
    (Not the oracle — we'll use check_accuracy.py for the real score)
    """
    print("\n" + "=" * 50)
    print("Self-accuracy check (on our own labelled images):")
    print("=" * 50)

    labelled_mask = image_labels >= 0
    true_labels   = image_labels[labelled_mask]
    pred_labels   = predicted_labels[labelled_mask]

    correct  = np.sum(true_labels == pred_labels)
    total    = len(true_labels)
    accuracy = correct / total

    print(f"Correct:  {correct} / {total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"(This is accuracy on OUR labels, not the true oracle accuracy)")

    return accuracy


def check_oracle_accuracy(predicted_labels):
    """
    Use the professor's check_accuracy function to get the TRUE accuracy.
    """
    print("\n" + "=" * 50)
    print("Checking TRUE accuracy (oracle)...")
    print("=" * 50)

    try:
        from check_accuracy import check_accuracy
        accuracy, n_correct, n_total = check_accuracy(predicted_labels)
        print(f"Oracle accuracy: {accuracy:.2%}")
        print(f"Correct:         {n_correct} / {n_total}")
        print(f"Target:          99.00%")
        print(f"Gap to target:   {(0.99 - accuracy)*100:.2f}%")
        return accuracy
    except Exception as e:
        print(f"Could not run oracle: {e}")
        return None


def save_results(svm, scaler, predicted_labels, margins, scores):
    """Save everything needed for Step 5."""
    # Save SVM model
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm, f)

    # Save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save predictions and margins
    np.save("predicted_labels.npy", predicted_labels)
    np.save("margins.npy",          margins)
    np.save("scores.npy",           scores)

    print("\n" + "=" * 50)
    print("Saved files:")
    print("  svm_model.pkl        ← trained SVM model")
    print("  scaler.pkl           ← feature normalizer")
    print("  predicted_labels.npy ← labels for all 10,000 images")
    print("  margins.npy          ← confidence score per image")
    print("  scores.npy           ← full decision scores")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data from Steps 1 and 3
    features, image_labels = load_data()

    # 2. Prepare and normalize training data
    (train_features_norm, train_labels,
     train_indices, all_features_norm,
     scaler, weights) = prepare_training_data(features, image_labels)

    # 3. Train the RBF-SVM
    svm = train_svm(train_features_norm, train_labels, weights)

    # 4. Predict labels for all 10,000 images
    predicted_labels, scores, margins = predict_all(
        svm, all_features_norm, image_labels)

    # 5. Check self-accuracy on our labelled images
    check_accuracy_on_labelled(predicted_labels, image_labels)

    # 6. Check true oracle accuracy
    oracle_acc = check_oracle_accuracy(predicted_labels)

    # 7. Save everything for Step 5
    save_results(svm, scaler, predicted_labels, margins, scores)

    print("\n" + "=" * 50)
    print("Step 4 COMPLETE ✓")
    if oracle_acc and oracle_acc >= 0.99:
        print("TARGET REACHED! Accuracy >= 99% already!")
    else:
        print("Next: Step 5 - Find boundary images & refine")
    print("=" * 50)