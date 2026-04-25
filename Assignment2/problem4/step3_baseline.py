import numpy as np
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
RESULTS_DIR = "./results"

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("Loading baseline vectors...")

train_vectors = np.load(f"{RESULTS_DIR}/train_baseline_vectors.npy")
test_vectors  = np.load(f"{RESULTS_DIR}/test_baseline_vectors.npy")
train_labels  = np.load(f"{RESULTS_DIR}/train_labels.npy")
test_labels   = np.load(f"{RESULTS_DIR}/test_labels.npy")

print(f"Train vectors shape : {train_vectors.shape}")
print(f"Test vectors shape  : {test_vectors.shape}")
print(f"Train labels shape  : {train_labels.shape}")
print(f"Test labels shape   : {test_labels.shape}")

# ─────────────────────────────────────────
# NORMALIZE
# ─────────────────────────────────────────
print("\nNormalizing vectors...")

# Fit scaler on train only then apply to both
scaler = StandardScaler()
train_vectors = scaler.fit_transform(train_vectors)
test_vectors  = scaler.transform(test_vectors)

print("Normalization done!")

# ─────────────────────────────────────────
# TRAIN CLASSIFIER
# ─────────────────────────────────────────
print("\nTraining SVM classifier...")

# Start training timer
train_start = time.time()

classifier = SVC(
    kernel = 'rbf',   # Radial Basis Function kernel
    C      = 10,      # regularization
    gamma  = 'scale'  # kernel coefficient
)
classifier.fit(train_vectors, train_labels)

# End training timer
train_end  = time.time()
train_time = (train_end - train_start) * 1000  # convert to ms

print(f"Training time: {train_time:.1f} ms")

# ─────────────────────────────────────────
# TEST CLASSIFIER
# ─────────────────────────────────────────
print("\nTesting classifier...")

# Start testing timer
test_start = time.time()

predictions = classifier.predict(test_vectors)

# End testing timer
test_end  = time.time()
test_time = (test_end - test_start) * 1000  # convert to ms

print(f"Testing time: {test_time:.1f} ms")

# ─────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────
accuracy = accuracy_score(test_labels, predictions) * 100

print("\n" + "="*50)
print("BASELINE RESULTS")
print("="*50)
print(f"Accuracy      : {accuracy:.1f}%")
print(f"Training time : {train_time:.1f} ms")
print(f"Testing time  : {test_time:.1f} ms")

# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────
print("\nGenerating confusion matrix...")

cm = confusion_matrix(test_labels, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot      = True,
    fmt        = 'd',
    cmap       = 'Blues',
    xticklabels= range(10),
    yticklabels= range(10)
)
plt.title(f'Baseline Classifier (Average MFCC)\nAccuracy: {accuracy:.1f}%')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/baseline_confusion_matrix.png")
plt.show()
print(f"Confusion matrix saved!")

# ─────────────────────────────────────────
# PER DIGIT ACCURACY
# ─────────────────────────────────────────
print("\nPer digit accuracy:")
print("-"*30)
for digit in range(10):
    # Get indices of this digit
    idx = np.where(test_labels == digit)[0]
    # Get predictions for this digit
    digit_preds = predictions[idx]
    digit_true  = test_labels[idx]
    # Calculate accuracy
    digit_acc = accuracy_score(digit_true, digit_preds) * 100
    print(f"  Digit {digit}: {digit_acc:.1f}%")

# ─────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────
np.save(f"{RESULTS_DIR}/baseline_predictions.npy", predictions)

# Save summary
summary = {
    'accuracy'     : accuracy,
    'train_time_ms': train_time,
    'test_time_ms' : test_time
}
np.save(f"{RESULTS_DIR}/baseline_summary.npy", summary)

print("\n" + "="*50)
print("BASELINE SUMMARY (save this for comparison table)")
print("="*50)
print(f"  Accuracy      : {accuracy:.1f}%")
print(f"  Training time : {train_time:.1f} ms")
print(f"  Testing time  : {test_time:.1f} ms")
print("\n✅ Step 3 Complete — Baseline recorded!")