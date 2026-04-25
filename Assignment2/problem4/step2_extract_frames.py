import os
import librosa
import numpy as np
from collections import Counter
import re

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
TRAIN_DIR   = "../data/train"
TEST_DIR    = "../data/test"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
FRAME_SIZE_MS  = 15      # ms
N_MFCC         = 13      # number of MFCC coefficients per frame

print(f"Frame size : {FRAME_SIZE_MS}ms (computed from each file sample rate)")
print(f"MFCC coefficients per frame: {N_MFCC}")

# ─────────────────────────────────────────
# FUNCTION: Split signal into 15ms frames
# ─────────────────────────────────────────
def split_into_frames(signal, frame_size):
    """
    Split signal into non-overlapping frames of frame_size samples
    Drop the last incomplete frame
    
    Input : signal shape [total_samples]
    Output: frames  shape [num_frames, frame_size]
    """
    num_frames = len(signal) // frame_size
    # Trim signal to fit exact number of frames
    signal = signal[:num_frames * frame_size]
    # Reshape into frames
    frames = signal.reshape(num_frames, frame_size)
    return frames

# ─────────────────────────────────────────
# FUNCTION: Extract MFCC from one frame
# ─────────────────────────────────────────
def extract_mfcc_from_frame(frame, sr, n_mfcc):
    """
    Extract MFCC features from a single frame
    
    Input : frame shape [frame_size]
    Output: mfcc  shape [n_mfcc]
    """
    # librosa expects float32
    frame = frame.astype(np.float32)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y      = frame,
        sr     = sr,
        n_mfcc = n_mfcc,
        n_fft  = len(frame),   # FFT size = frame size
        hop_length = len(frame) # no overlap
    )
    # mfcc shape: [n_mfcc, 1] → squeeze to [n_mfcc]
    return mfcc[:, 0]

# ─────────────────────────────────────────
# FUNCTION: Process all files in a folder
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


def process_folder(folder):
    """
    Process all wav files in folder
    Returns:
        all_mfcc_frames : list of arrays, each shape [num_frames, n_mfcc]
        all_labels      : list of int labels
        all_filenames   : list of filenames
    """
    all_mfcc_frames = []
    all_labels      = []
    all_filenames   = []
    sample_rates    = Counter()

    files = sorted(os.listdir(folder))
    total = len([f for f in files if f.lower().endswith(".wav")])
    count = 0

    for filename in files:
        if not filename.lower().endswith(".wav"):
            continue

        count += 1
        if count % 100 == 0:
            print(f"  Processing {count}/{total}...")

        # Extract label from filename
        label    = extract_label_from_filename(filename)
        filepath = os.path.join(folder, filename)

        # Load wav
        signal, sr = librosa.load(filepath, sr=None)
        sample_rates[sr] += 1

        # 15ms frame size in samples for this file's sample rate
        frame_size = int(sr * FRAME_SIZE_MS / 1000)
        if frame_size <= 0:
            raise ValueError(f"Invalid frame size for {filename} at sample rate {sr}")

        # Split into 15ms frames
        frames = split_into_frames(signal, frame_size)
        if len(frames) == 0:
            raise ValueError(
                f"Utterance too short after framing (filename={filename}, frame_size={frame_size})"
            )

        # Extract MFCC from each frame
        mfcc_frames = []
        for frame in frames:
            mfcc = extract_mfcc_from_frame(frame, sr, N_MFCC)
            mfcc_frames.append(mfcc)

        # Stack into matrix [num_frames, n_mfcc]
        mfcc_frames = np.array(mfcc_frames)

        all_mfcc_frames.append(mfcc_frames)
        all_labels.append(label)
        all_filenames.append(filename)

    return all_mfcc_frames, all_labels, all_filenames, sample_rates

# ─────────────────────────────────────────
# PROCESS TRAIN & TEST
# ─────────────────────────────────────────
print("\nProcessing training data...")
train_frames, train_labels, train_files, train_rates = process_folder(TRAIN_DIR)

print("\nProcessing test data...")
test_frames, test_labels, test_files, test_rates = process_folder(TEST_DIR)

print("\nDetected sample rates (train):")
for sr, n in sorted(train_rates.items()):
    frame_samples = int(sr * FRAME_SIZE_MS / 1000)
    print(f"  {sr} Hz: {n} files -> 15ms = {frame_samples} samples")

print("Detected sample rates (test):")
for sr, n in sorted(test_rates.items()):
    frame_samples = int(sr * FRAME_SIZE_MS / 1000)
    print(f"  {sr} Hz: {n} files -> 15ms = {frame_samples} samples")

# ─────────────────────────────────────────
# EXPLORE FRAMES
# ─────────────────────────────────────────
frame_counts = [len(f) for f in train_frames]
print("\n" + "="*50)
print("FRAME STATISTICS")
print("="*50)
print(f"Min frames per utterance : {min(frame_counts)}")
print(f"Max frames per utterance : {max(frame_counts)}")
print(f"Avg frames per utterance : {np.mean(frame_counts):.1f}")
print(f"MFCC shape per utterance example: {train_frames[0].shape}")

MAX_FRAMES = max(frame_counts)
INPUT_SIZE = MAX_FRAMES * N_MFCC
print(f"\nMax frames (padding target) : {MAX_FRAMES}")
print(f"AE input size (flattened)   : {MAX_FRAMES} x {N_MFCC} = {INPUT_SIZE}")

# ─────────────────────────────────────────
# BASELINE VECTORS — Average all frames
# ─────────────────────────────────────────
print("\nCreating baseline vectors (average frames)...")

def make_baseline_vectors(all_mfcc_frames):
    """
    Average all frames per utterance → single vector [n_mfcc]
    """
    vectors = []
    for mfcc_frames in all_mfcc_frames:
        avg = np.mean(mfcc_frames, axis=0)  # shape [n_mfcc]
        vectors.append(avg)
    return np.array(vectors)

train_baseline = make_baseline_vectors(train_frames)
test_baseline  = make_baseline_vectors(test_frames)

print(f"Baseline train shape: {train_baseline.shape}")
print(f"Baseline test shape : {test_baseline.shape}")

# ─────────────────────────────────────────
# AE VECTORS — Zero pad + Flatten
# ─────────────────────────────────────────
print("\nCreating AE vectors (zero-pad + flatten)...")

def make_ae_vectors(all_mfcc_frames, max_frames, n_mfcc):
    """
    Zero-pad each utterance to max_frames then flatten
    Input : list of [num_frames, n_mfcc] arrays
    Output: array of shape [num_utterances, max_frames * n_mfcc]
    """
    vectors = []
    for mfcc_frames in all_mfcc_frames:
        num_frames = len(mfcc_frames)
        # Create zero matrix of max size
        padded = np.zeros((max_frames, n_mfcc))
        # Fill with actual frames
        padded[:num_frames, :] = mfcc_frames
        # Flatten to 1D vector
        flat = padded.flatten()
        vectors.append(flat)
    return np.array(vectors)

train_ae = make_ae_vectors(train_frames, MAX_FRAMES, N_MFCC)
test_ae  = make_ae_vectors(test_frames,  MAX_FRAMES, N_MFCC)

print(f"AE train shape: {train_ae.shape}")
print(f"AE test shape : {test_ae.shape}")

# ─────────────────────────────────────────
# SAVE EVERYTHING
# ─────────────────────────────────────────
print("\nSaving all arrays...")

# Raw frames (variable length - saved as object array)
np.save(f"{RESULTS_DIR}/train_mfcc_frames.npy",
        np.array(train_frames, dtype=object))
np.save(f"{RESULTS_DIR}/test_mfcc_frames.npy",
        np.array(test_frames, dtype=object))

# Labels
np.save(f"{RESULTS_DIR}/train_labels.npy", np.array(train_labels))
np.save(f"{RESULTS_DIR}/test_labels.npy",  np.array(test_labels))

# Baseline vectors
np.save(f"{RESULTS_DIR}/train_baseline_vectors.npy", train_baseline)
np.save(f"{RESULTS_DIR}/test_baseline_vectors.npy",  test_baseline)

# AE vectors
np.save(f"{RESULTS_DIR}/train_ae_vectors.npy", train_ae)
np.save(f"{RESULTS_DIR}/test_ae_vectors.npy",  test_ae)

# Save max_frames for later use
np.save(f"{RESULTS_DIR}/max_frames.npy", np.array([MAX_FRAMES]))

print("\nSaved files:")
for f in os.listdir(RESULTS_DIR):
    path = os.path.join(RESULTS_DIR, f)
    size = os.path.getsize(path) / 1024
    print(f"  {f:40s} {size:.1f} KB")

print("\n✅ Step 2 Complete — Frames extracted successfully!")