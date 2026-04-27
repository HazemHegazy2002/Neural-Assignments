"""
diagnose_speakers.py
--------------------
Analyzes the dataset to answer:
1. What is the speaker ID pattern in filenames?
2. Do the same speakers appear in both Train and Test?
3. How many unique speakers are there?
4. Does each speaker record all digits or just some?

Run this BEFORE training to understand your data split quality.
"""

import os
from collections import defaultdict

# ── Set your paths ────────────────────────────────────────────────────────────
TRAIN_DIR = r"D:\My life\EECE\4th year\2nd term\Neural Networks\Project\audio-dataset\Train"   # e.g. "data/train"
TEST_DIR  = r"D:\My life\EECE\4th year\2nd term\Neural Networks\Project\audio-dataset\Test"    # e.g. "data/test"

SUPPORTED_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")

def extract_speaker(stem: str) -> str:
    """
    Given a filename stem like 'M20n_6', return everything before the last '_'.
    e.g.  'M20n_6'   → 'M20n'
          'F05b_3'   → 'F05b'
          'abc_xy_9' → 'abc_xy'
    """
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem

def scan_dir(root_dir: str) -> dict:
    """
    Returns a dict:  speaker_id → set of digits they recorded
    """
    speaker_digits = defaultdict(set)
    for fname in os.listdir(root_dir):
        if not fname.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
        stem      = os.path.splitext(fname)[0]   # e.g. 'M20n_6'
        last_char = stem[-1]
        if not last_char.isdigit():
            continue
        speaker = extract_speaker(stem)
        speaker_digits[speaker].add(int(last_char))
    return dict(speaker_digits)

# ── Scan both directories ─────────────────────────────────────────────────────
print("=" * 60)
print("SCANNING TRAIN DIR ...")
train_speakers = scan_dir(TRAIN_DIR)
print(f"  Unique speakers : {len(train_speakers)}")
print(f"  Sample speaker IDs: {list(train_speakers.keys())[:10]}")

print("\nSCANNING TEST DIR ...")
test_speakers = scan_dir(TEST_DIR)
print(f"  Unique speakers : {len(test_speakers)}")
print(f"  Sample speaker IDs: {list(test_speakers.keys())[:10]}")

# ── Overlap analysis ──────────────────────────────────────────────────────────
overlap = set(train_speakers.keys()) & set(test_speakers.keys())
print("\n" + "=" * 60)
print(f"SPEAKER OVERLAP BETWEEN TRAIN AND TEST:")
print(f"  Speakers in BOTH train and test : {len(overlap)}")

if overlap:
    print(f"  ⚠️  LEAKAGE DETECTED — {len(overlap)} speakers appear in both sets.")
    print(f"  Example overlapping speakers: {list(overlap)[:10]}")
else:
    print(f"  ✅ No overlap — train and test are speaker-independent.")

# ── Per-speaker digit coverage ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAIN — digits recorded per speaker (first 10 speakers):")
for spk, digits in list(train_speakers.items())[:10]:
    print(f"  {spk:15s} → digits: {sorted(digits)}")

print("\nTEST — digits recorded per speaker (first 10 speakers):")
for spk, digits in list(test_speakers.items())[:10]:
    print(f"  {spk:15s} → digits: {sorted(digits)}")

# ── Recommendation ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RECOMMENDATION:")
if overlap:
    print("  → Use speaker-aware validation split in train.py.")
    print("    All recordings from one speaker must stay in train OR val, not both.")
    print("    This prevents the model from learning speaker identity instead of digits.")
else:
    print("  → Your train/test split is already speaker-independent.")
    print("    The 100% val accuracy issue may be due to the small val set size (180 samples).")
    print("    Consider increasing VAL_SPLIT or checking for other leakage sources.")