import os
import librosa
import numpy as np
from collections import Counter
import re

# ─────────────────────────────────────────
# PATHS  
# ─────────────────────────────────────────
TRAIN_DIR = "../data/train"
TEST_DIR  = "../data/test"

# ─────────────────────────────────────────
# FUNCTION: Load all wav files from a folder
# ─────────────────────────────────────────
def extract_label_from_filename(filename):
    """
    Parse digit label in a filename.

    Supported patterns include:
      - old: 0_02_41.wav  -> 0
      - new: C03n_0.wav   -> 0
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")

    if parts and parts[0].isdigit() and 0 <= int(parts[0]) <= 9:
        return int(parts[0])

    if parts and parts[-1].isdigit() and 0 <= int(parts[-1]) <= 9:
        return int(parts[-1])

    single_digit_tokens = [p for p in parts if len(p) == 1 and p.isdigit()]
    if len(single_digit_tokens) == 1:
        return int(single_digit_tokens[0])

    match = re.search(r"([0-9])$", stem)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not parse digit label from filename: {filename}")


def load_data(folder):
    data   = []   # will hold (signal, sample_rate, label)
    labels = []

    files = sorted(os.listdir(folder))

    for filename in files:
        if not filename.lower().endswith(".wav"):
            continue

        # Extract label from filename
        label = extract_label_from_filename(filename)

        filepath = os.path.join(folder, filename)

        # Load wav file
        signal, sr = librosa.load(filepath, sr=None)  
        # sr=None keeps the original sample rate

        data.append((signal, sr, filename))
        labels.append(label)

    return data, labels

# ─────────────────────────────────────────
# LOAD TRAIN & TEST
# ─────────────────────────────────────────
print("Loading training data...")
train_data, train_labels = load_data(TRAIN_DIR)

print("Loading test data...")
test_data, test_labels = load_data(TEST_DIR)

# ─────────────────────────────────────────
# EXPLORE & VERIFY
# ─────────────────────────────────────────
print("\n" + "="*50)
print("DATASET SUMMARY")
print("="*50)

print(f"\nTotal train files : {len(train_data)}")
print(f"Total test files  : {len(test_data)}")

# Count per digit
train_counts = Counter(train_labels)
test_counts  = Counter(test_labels)

print("\nTrain samples per digit:")
for digit in sorted(train_counts):
    print(f"  Digit {digit}: {train_counts[digit]} files")

print("\nTest samples per digit:")
for digit in sorted(test_counts):
    print(f"  Digit {digit}: {test_counts[digit]} files")

# Check sample rates & durations
print("\nSample file info:")
for i in range(min(3, len(train_data))):  # show first 3 files
    signal, sr, fname = train_data[i]
    duration = len(signal) / sr
    print(f"  {fname}")
    print(f"    Sample rate : {sr} Hz")
    print(f"    Duration    : {duration:.3f} sec")
    print(f"    Samples     : {len(signal)}")

# Check min/max duration across all files
durations = [len(s) / sr for s, sr, _ in train_data]
print(f"\nTrain durations:")
print(f"  Min : {min(durations):.3f} sec")
print(f"  Max : {max(durations):.3f} sec")
print(f"  Avg : {np.mean(durations):.3f} sec")

print("\n✅ Step 1 Complete — Data loaded successfully!")