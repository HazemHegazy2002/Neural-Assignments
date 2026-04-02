# =============================================================
#  Pipeline 2 - Step 3: Initial SVM Training (SVM-1)
#  Assignment 1 - Part 3
# =============================================================
#
#  Trains a multi-class SVM (RBF kernel, one-vs-one) on the
#  2400 training images (300 seed + 2100 augmented) with weights.
#
#  Then predicts labels + confidence scores for ALL 10,000 images.
#
#  Outputs:
#    svm_model.pkl         — trained SVM model
#    predicted_labels.npy  — predicted label for all 10,000 images
#    decision_scores.npy   — raw decision scores  (10000, n_classes)
#    margins.npy           — confidence margin for each image (10000,)
#                            margin = top_score - second_score
#
#  Also calls check_accuracy to report the first accuracy reading.
# =============================================================

import numpy as np
import pickle
import time
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add folder containing check_accuracy.py to path (same directory as this script)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_accuracy import check_accuracy

# ── CONFIG ────────────────────────────────────────────────────
SVM_C      = 10.0    # regularisation parameter
SVM_GAMMA  = 'scale' # RBF kernel bandwidth
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────────────


def load_training_data():
    train_images  = np.load("train_images.npy")   # (2400, 784)
    train_labels  = np.load("train_labels.npy")   # (2400,)
    train_weights = np.load("train_weights.npy")  # (2400,)
    all_images    = np.load("all_images.npy")      # (10000, 784)

    print(f"Training set : {train_images.shape}")
    print(f"Full dataset : {all_images.shape}")
    return train_images, train_labels, train_weights, all_images


def train_svm(train_images, train_labels, train_weights):
    """
    Train RBF-SVM with sample weights using sklearn.
    Uses StandardScaler + SVC (decision_function_shape='ovo').
    """
    print("\nTraining SVM-1  (RBF kernel, one-vs-one, C={}, gamma={}) ...".format(
        SVM_C, SVM_GAMMA))
    print("  This may take 1-3 minutes ...")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_images)

    svm = SVC(
        kernel='rbf',
        C=SVM_C,
        gamma=SVM_GAMMA,
        decision_function_shape='ovo',   # one-vs-one
        random_state=RANDOM_STATE,
        cache_size=2000                  # MB — speeds up training
    )

    t0 = time.time()
    svm.fit(X_train, train_labels, sample_weight=train_weights)
    elapsed = time.time() - t0

    print(f"  Training done in {elapsed:.1f} s")
    return svm, scaler


def predict_all(svm, scaler, all_images):
    """
    Predict labels and decision scores for all 10,000 images.
    Returns predicted_labels (10000,), scores (10000, K*(K-1)/2),
    and margins (10000,).
    """
    print("\nPredicting labels for all 10,000 images ...")
    t0 = time.time()

    X_all = scaler.transform(all_images)
    predicted_labels = svm.predict(X_all)               # (10000,)
    scores           = svm.decision_function(X_all)     # (10000, n_classifiers)

    elapsed = time.time() - t0
    print(f"  Prediction done in {elapsed:.1f} s")

    # ── Compute margin = difference between top-2 scores ──
    # For OVO, each row has scores for each binary classifier.
    # We use the two highest values as our confidence measure.
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]    # descending
    margins = sorted_scores[:, 0] - sorted_scores[:, 1] # (10000,)

    return predicted_labels, scores, margins


def save_results(svm, scaler, predicted_labels, scores, margins):
    # Save model + scaler together
    with open("svm_model.pkl", "wb") as f:
        pickle.dump({'svm': svm, 'scaler': scaler}, f)

    np.save("predicted_labels.npy", predicted_labels)
    np.save("decision_scores.npy",  scores)
    np.save("margins.npy",          margins)

    print("\nSaved:")
    for f in ["svm_model.pkl", "predicted_labels.npy",
              "decision_scores.npy", "margins.npy"]:
        print(f"  {f}")


def print_summary(predicted_labels, margins, accuracy, n_correct, n_total):
    print("\n" + "═" * 52)
    print("  STEP 3 SUMMARY  —  SVM-1 Results")
    print("═" * 52)
    print(f"  Kernel         : RBF  (C={SVM_C}, gamma={SVM_GAMMA})")
    print(f"  Training images: 2400  (300 seed + 2100 augmented)")
    print(f"  Accuracy       : {accuracy*100:.2f}%  ({n_correct}/{n_total})")
    print(f"  Margin stats:")
    print(f"    Min    : {margins.min():.4f}")
    print(f"    Mean   : {margins.mean():.4f}")
    print(f"    Median : {np.median(margins):.4f}")
    print(f"    Max    : {margins.max():.4f}")
    print(f"    75th % : {np.percentile(margins, 75):.4f}")

    print("\n  Predicted label distribution:")
    unique, counts = np.unique(predicted_labels, return_counts=True)
    for d, c in zip(unique, counts):
        print(f"    Digit {d}: {c:5d}")
    print("═" * 52)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # 1. Load data
    train_images, train_labels, train_weights, all_images = load_training_data()

    # 2. Train SVM-1
    svm, scaler = train_svm(train_images, train_labels, train_weights)

    # 3. Predict all 10,000 images
    predicted_labels, scores, margins = predict_all(svm, scaler, all_images)

    # 4. Check accuracy using oracle
    print("\nChecking accuracy with oracle ...")
    accuracy, n_correct, n_total = check_accuracy(predicted_labels)
    print(f"  Accuracy: {accuracy*100:.2f}%  ({n_correct}/{n_total} correct)")

    # 5. Save everything
    save_results(svm, scaler, predicted_labels, scores, margins)

    # 6. Summary
    print_summary(predicted_labels, margins, accuracy, n_correct, n_total)

    print("\nStep 3 done. Run step4_active_refinement.py next.")