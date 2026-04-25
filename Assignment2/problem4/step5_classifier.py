import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RESULTS_DIR = "./results"

SVM_PARAM_GRID = {
    "svm__kernel": ["rbf"],
    "svm__C": [3, 10, 30, 100, 300],
    "svm__gamma": [0.0005, 0.001, 0.003, 0.005, "scale"],
    "svm__class_weight": [None, "balanced"],
}


def load_baseline_accuracy(summary_path):
    """Load baseline accuracy from the Step 3 summary file if available."""
    if not os.path.exists(summary_path):
        return None

    try:
        summary_obj = np.load(summary_path, allow_pickle=True)
        if hasattr(summary_obj, "item"):
            summary_dict = summary_obj.item()
            if isinstance(summary_dict, dict) and "accuracy" in summary_dict:
                return float(summary_dict["accuracy"])
    except Exception:
        return None

    return None


def save_confusion_matrix(cm, accuracy, out_path, feature_mode):
    """Create and save a confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = np.arange(10)
    ax.set(
        xticks=classes,
        yticks=classes,
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=(
            f"AE + SVM ({feature_mode}) Confusion Matrix "
            f"(Accuracy: {accuracy:.1f}%)"
        ),
    )

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 5: Train an SVM classifier using strict AE bottleneck vectors "
            "and choose hyperparameters by cross-validation."
        )
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for GridSearchCV hyperparameter tuning.",
    )
    return parser.parse_args()


def train_svm_with_cv(train_vectors, train_labels, cv_folds):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC()),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=SVM_PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(train_vectors, train_labels)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def main():
    args = parse_args()

    print("Loading bottleneck vectors and labels...")

    train_vectors = np.load(f"{RESULTS_DIR}/train_bottleneck.npy")
    test_vectors = np.load(f"{RESULTS_DIR}/test_bottleneck.npy")
    train_labels = np.load(f"{RESULTS_DIR}/train_labels.npy")
    test_labels = np.load(f"{RESULTS_DIR}/test_labels.npy")

    print(f"Train bottleneck shape : {train_vectors.shape}")
    print(f"Test bottleneck shape  : {test_vectors.shape}")
    print(f"Train labels shape     : {train_labels.shape}")
    print(f"Test labels shape      : {test_labels.shape}")

    print("\nUsing strict AE-only features from Step 4.")
    print(f"Feature shape: {train_vectors.shape}")

    print("\nTraining CV-tuned SVM classifier...")
    train_start = time.time()

    classifier, best_params, best_cv = train_svm_with_cv(
        train_vectors, train_labels, args.cv_folds
    )
    print(
        f"CV accuracy: {best_cv * 100:.2f}% | "
        f"best params: {best_params}"
    )

    train_time_ms = (time.time() - train_start) * 1000
    print(f"Training time: {train_time_ms:.1f} ms")
    print(f"Selected train shape : {train_vectors.shape}")
    print(f"Selected test shape  : {test_vectors.shape}")

    print("\nTesting classifier...")
    test_start = time.time()
    predictions = classifier.predict(test_vectors)
    test_time_ms = (time.time() - test_start) * 1000
    print(f"Testing time: {test_time_ms:.1f} ms")

    accuracy = accuracy_score(test_labels, predictions) * 100.0

    print("\n" + "=" * 50)
    print("STEP 5 RESULTS (AE BOTTLENECK + SVM)")
    print("=" * 50)
    print("Feature mode  : strict (AE bottleneck only)")
    print(f"Best CV score : {best_cv * 100:.2f}%")
    print(f"Accuracy      : {accuracy:.1f}%")
    print(f"Training time : {train_time_ms:.1f} ms")
    print(f"Testing time  : {test_time_ms:.1f} ms")

    print("\nPer digit accuracy:")
    print("-" * 30)
    for digit in range(10):
        idx = np.where(test_labels == digit)[0]
        digit_acc = accuracy_score(test_labels[idx], predictions[idx]) * 100.0
        print(f"  Digit {digit}: {digit_acc:.1f}%")

    cm = confusion_matrix(test_labels, predictions, labels=np.arange(10))
    cm_path = f"{RESULTS_DIR}/ae_svm_confusion_matrix.png"
    save_confusion_matrix(cm, accuracy, cm_path, "strict")
    print(f"\nConfusion matrix saved: {cm_path}")

    summary = {
        "method": "ae_bottleneck_svm_cv",
        "feature_mode": "strict",
        "feature_description": "AE bottleneck only",
        "cv_folds": int(args.cv_folds),
        "cv_accuracy": float(best_cv * 100.0),
        "accuracy": float(accuracy),
        "train_time_ms": float(train_time_ms),
        "test_time_ms": float(test_time_ms),
        "svm_kernel": best_params["svm__kernel"],
        "svm_C": float(best_params["svm__C"]),
        "svm_gamma": best_params["svm__gamma"],
        "svm_class_weight": best_params["svm__class_weight"],
    }
    np.save(f"{RESULTS_DIR}/ae_svm_summary.npy", summary)
    np.save(f"{RESULTS_DIR}/ae_svm_predictions.npy", predictions)

    baseline_acc = load_baseline_accuracy(f"{RESULTS_DIR}/baseline_summary.npy")
    if baseline_acc is not None:
        delta = accuracy - baseline_acc
        comparison = {
            "baseline_accuracy": float(baseline_acc),
            "ae_svm_accuracy": float(accuracy),
            "absolute_gain": float(delta),
        }
        np.save(f"{RESULTS_DIR}/comparison_baseline_vs_ae_svm.npy", comparison)

        print("\nBaseline comparison:")
        print("-" * 30)
        print(f"Baseline accuracy : {baseline_acc:.1f}%")
        print(f"AE + SVM accuracy : {accuracy:.1f}%")
        print(f"Gain              : {delta:+.1f}%")
    else:
        print("\nBaseline summary not found; skipped automatic comparison.")

    print("\nSaved files:")
    print(f"  {RESULTS_DIR}/ae_svm_summary.npy")
    print(f"  {RESULTS_DIR}/ae_svm_predictions.npy")
    print(f"  {RESULTS_DIR}/ae_svm_confusion_matrix.png")
    if baseline_acc is not None:
        print(f"  {RESULTS_DIR}/comparison_baseline_vs_ae_svm.npy")

    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()