"""
Pipeline 1 - Step 6: Retrain SVM & Check Accuracy
===================================================
What this step does:
- Loads updated labels from Step 5 (with human corrections)
- Loads sample weights (trusted=100, cluster=1)
- Retrains the RBF-SVM with weighted samples
- Predicts all 10,000 images again
- Recomputes margins for next iteration
- Checks oracle accuracy
- Tells you if you need another Step 5 → Step 6 iteration

After running this:
- If accuracy >= 99% → DONE!
- If accuracy < 99%  → run Step 5 again, then Step 6 again
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import time
import os
import pickle

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TARGET_ACCURACY = 0.99    # 99%
C_GRID = [10, 30, 100]
GAMMA_GRID = ['scale', 0.01, 0.005]


def load_data():
    """Load everything needed for retraining."""
    print("=" * 50)
    print("Loading data from previous steps...")
    print("=" * 50)

    for f in ["hog_features.npy", "image_labels.npy", "sample_weights.npy"]:
        if not os.path.exists(f):
            print(f"ERROR: '{f}' not found!")
            print("Make sure you ran Step 5 first.")
            exit()

    features       = np.load("hog_features.npy")      # (10000, 324)
    image_labels   = np.load("image_labels.npy")      # (10000,)
    sample_weights = np.load("sample_weights.npy")    # (10000,)
    trusted_indices = []

    if os.path.exists("trusted_indices.npy"):
        trusted_indices = list(np.load("trusted_indices.npy"))

    print(f"Features shape:         {features.shape}")
    print(f"Total images:           {len(image_labels)}")
    print(f"Labelled images:        {np.sum(image_labels >= 0)}")
    print(f"Unknown images (-1):    {np.sum(image_labels == -1)}")
    print(f"Trusted human labels:   {len(trusted_indices)} "
          f"(weight=100)")
    print(f"Cluster labels:         "
          f"{np.sum(image_labels >= 0) - len(trusted_indices)} "
          f"(weight=1)")

    return features, image_labels, sample_weights, trusted_indices


def prepare_training_data(features, image_labels, sample_weights):
    """
    Prepare training data with weights.

    Key difference from Step 4:
    - Step 4: all weights = 1
    - Step 6: trusted images weight = 100, cluster images weight = 1

    This means the SVM will prioritize getting the
    human-verified images correct over cluster-labelled ones.
    """
    print("\n" + "=" * 50)
    print("Preparing weighted training data...")
    print("=" * 50)

    # Only use labelled images (skip -1 unknown)
    labelled_mask = image_labels >= 0

    train_features = features[labelled_mask]
    train_labels   = image_labels[labelled_mask]
    train_weights  = sample_weights[labelled_mask]

    print(f"Training samples:       {len(train_features)}")
    print(f"High-weight samples:    "
          f"{np.sum(train_weights == 100)} (human verified)")
    print(f"Normal-weight samples:  "
          f"{np.sum(train_weights == 1)} (cluster labelled)")

    # Refit scaler every iteration so normalization matches current labels.
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    all_features_norm   = scaler.transform(features)
    print("Scaler refit on current labelled data ✓")

    return train_features_norm, train_labels, train_weights, all_features_norm, scaler


def retrain_svm(train_features, train_labels, train_weights):
    """
    Retrain the SVM with sample weights.

    The sample_weight parameter tells the SVM:
    'Pay 100x more attention to getting these
     human-verified images correct'

    This is the key improvement over Step 4:
    - Mistakes on trusted images are penalized heavily
    - The SVM boundary shifts to respect human corrections
    """
    print("\n" + "=" * 50)
    print("Retraining RBF-SVM with weighted samples...")
    print("This may take 2-5 minutes, please wait...")
    print("=" * 50)

    start = time.time()

    # Small validation split to pick better C/gamma quickly.
    try:
        X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
            train_features, train_labels, train_weights,
            test_size=0.15, random_state=42, stratify=train_labels
        )
    except ValueError:
        X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
            train_features, train_labels, train_weights,
            test_size=0.15, random_state=42
        )

    best_score = -1.0
    best_params = None

    print("Hyperparameter search (quick grid):")
    for c_value in C_GRID:
        for gamma_value in GAMMA_GRID:
            candidate = SVC(
                kernel='rbf',
                C=c_value,
                gamma=gamma_value,
                decision_function_shape='ovr',
                random_state=42
            )
            candidate.fit(X_fit, y_fit, sample_weight=w_fit)
            val_pred = candidate.predict(X_val)
            val_score = float(np.mean(val_pred == y_val))
            print(f"  C={c_value:<3} gamma={gamma_value:<6} val_acc={val_score:.4f}")

            if val_score > best_score:
                best_score = val_score
                best_params = (c_value, gamma_value)

    print(f"Best params: C={best_params[0]}, gamma={best_params[1]} "
          f"(val_acc={best_score:.4f})")

    # Retrain best model on all labelled data.
    svm = SVC(
        kernel='rbf',
        C=best_params[0],
        gamma=best_params[1],
        decision_function_shape='ovr',
        random_state=42
    )
    svm.fit(train_features, train_labels, sample_weight=train_weights)

    elapsed = time.time() - start
    print(f"SVM retrained in {elapsed:.1f} seconds ✓")

    return svm


def predict_and_compute_margins(svm, all_features_norm):
    """
    Predict all 10,000 images and compute new margins.
    These margins will be used in the NEXT Step 5 iteration
    to find the next set of boundary images.
    """
    print("\n" + "=" * 50)
    print("Predicting all 10,000 images...")
    print("=" * 50)

    start = time.time()

    predicted_labels = svm.predict(all_features_norm)
    scores           = svm.decision_function(all_features_norm)

    elapsed = time.time() - start
    print(f"Prediction done in {elapsed:.1f} seconds ✓")

    # Compute new margins
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    margins       = sorted_scores[:, 0] - sorted_scores[:, 1]

    print(f"\nNew margin statistics:")
    print(f"  Average margin: {margins.mean():.3f}")
    print(f"  Min margin:     {margins.min():.3f}")
    print(f"  Max margin:     {margins.max():.3f}")

    print(f"\nPrediction distribution:")
    for digit in range(10):
        count = np.sum(predicted_labels == digit)
        bar   = "█" * (count // 100)
        print(f"  Digit {digit}: {count:5d} images  {bar}")

    return predicted_labels, scores, margins


def check_oracle_accuracy(predicted_labels):
    """Check true accuracy using professor's oracle."""
    print("\n" + "=" * 50)
    print("Checking TRUE accuracy (oracle)...")
    print("=" * 50)

    try:
        from check_accuracy import check_accuracy
        accuracy, n_correct, n_total = check_accuracy(predicted_labels)

        print(f"Oracle accuracy:  {accuracy:.2%}")
        print(f"Correct:          {n_correct} / {n_total}")
        print(f"Target:           {TARGET_ACCURACY:.2%}")

        if accuracy >= TARGET_ACCURACY:
            print(f"TARGET REACHED! ✅")
        else:
            gap         = (TARGET_ACCURACY - accuracy) * 10000
            print(f"Still need to fix ~{gap:.0f} more images")
            print(f"→ Run Step 5 again, then Step 6 again")

        return accuracy, n_correct

    except Exception as e:
        print(f"Could not run oracle: {e}")
        return None, None


def show_confusion_matrix(predicted_labels, image_labels):
    """
    Show confusion matrix on our labelled images.
    Rows = true label, Columns = predicted label
    Diagonal = correct predictions
    Off-diagonal = mistakes
    """
    print("\n" + "=" * 50)
    print("Confusion matrix (on labelled images):")
    print("=" * 50)

    labelled_mask = image_labels >= 0
    true_labels   = image_labels[labelled_mask]
    pred_labels   = predicted_labels[labelled_mask]

    cm = confusion_matrix(true_labels, pred_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(range(10)))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title("Confusion Matrix — Step 6 Retrained SVM\n"
                 "Rows=True Label, Cols=Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.show()
    print("Confusion matrix saved as 'confusion_matrix.png'")


def save_results(svm, scaler, predicted_labels, margins, scores):
    """Save updated model and predictions for next iteration."""
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save("predicted_labels.npy", predicted_labels)
    np.save("margins.npy",          margins)
    np.save("scores.npy",           scores)

    print("\n" + "=" * 50)
    print("Saved updated files:")
    print("  svm_model.pkl        ← retrained SVM")
    print("  scaler.pkl           ← feature normalizer")
    print("  predicted_labels.npy ← new predictions")
    print("  margins.npy          ← new confidence scores")
    print("  scores.npy           ← new decision scores")


def print_iteration_summary(accuracy, n_correct, trusted_indices):
    """Print a clear summary of where we are."""
    print("\n" + "=" * 50)
    print("ITERATION SUMMARY")
    print("=" * 50)
    print(f"Current accuracy:     {accuracy:.2%}")
    print(f"Correct images:       {n_correct} / 10000")
    print(f"Human labels used:    {len(trusted_indices)}")
    print(f"Manual time so far:   "
          f"{len(trusted_indices) * 10}s "
          f"≈ {len(trusted_indices) * 10 / 60:.1f} minutes")

    if accuracy >= TARGET_ACCURACY:
        print("\n" + "🎉 " * 10)
        print("PIPELINE 1 COMPLETE!")
        print(f"Final accuracy: {accuracy:.2%}")
        print(f"Total human labels: {len(trusted_indices)}")
        print(f"Total manual time: "
              f"{len(trusted_indices) * 10 / 60:.1f} min "
              f"(vs 27.8 hours manual)")
        print("🎉 " * 10)
    else:
        gap = (TARGET_ACCURACY - accuracy) * 100
        print(f"\nGap to target: {gap:.2f}%")
        print("\nNext steps:")
        print("  1. Run step5_boundary_images.py again")
        print("  2. Run step6_retrain_svm.py again")
        print("  Repeat until accuracy >= 99%")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load updated data from Step 5
    features, image_labels, sample_weights, trusted_indices = load_data()

    # 2. Prepare weighted training data
    (train_features_norm, train_labels,
     train_weights, all_features_norm,
     scaler) = prepare_training_data(features, image_labels, sample_weights)

    # 3. Retrain SVM with weights
    svm = retrain_svm(train_features_norm, train_labels, train_weights)

    # 4. Predict all images + compute new margins
    predicted_labels, scores, margins = predict_and_compute_margins(
        svm, all_features_norm)

    # 5. Check oracle accuracy
    accuracy, n_correct = check_oracle_accuracy(predicted_labels)

    # 6. Show confusion matrix
    show_confusion_matrix(predicted_labels, image_labels)

    # 7. Save everything for next iteration
    save_results(svm, scaler, predicted_labels, margins, scores)

    # 8. Print summary and next steps
    if accuracy:
        print_iteration_summary(accuracy, n_correct, trusted_indices)

    print("\n" + "=" * 50)
    print("Step 6 COMPLETE ✓")
    print("=" * 50)