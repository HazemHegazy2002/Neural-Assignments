"""
dataset.py
----------
Handles all data loading and preprocessing for the Arabic digit speech recognition system.

Responsibilities:
    - Walk TRAIN_DIR / TEST_DIR and collect (filepath, label) pairs.
    - Load each audio file with librosa.
    - Convert the raw waveform to a log-mel spectrogram.
    - Resize the spectrogram to a fixed shape and normalise to [0, 1].
    - Expose a PyTorch Dataset so DataLoader can batch and shuffle the data.

Designed to be easily modified for parts (b), (c), and (d):
    - Audio augmentation (speed, noise) can be applied inside _load_waveform().
    - Image augmentation (squeeze, expand, noise) can be applied inside _to_spectrogram()
      or via torchvision transforms passed to SpeechDataset.
"""

import os
import sys
import numpy as np


def _ensure_project_venv() -> None:
    """Re-run this script with the repo venv Python when available on Windows."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    venv_python = os.path.join(repo_root, ".venv", "Scripts", "python.exe")

    if os.name != "nt":
        return
    if not os.path.isfile(venv_python):
        return
    if os.path.normcase(os.path.abspath(sys.executable)) == os.path.normcase(os.path.abspath(venv_python)):
        return

    script_path = os.path.abspath(__file__)
    os.execv(venv_python, [venv_python, script_path, *sys.argv[1:]])


_ensure_project_venv()

import librosa
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────
#  CONFIG  –  adjust paths here before running
# ─────────────────────────────────────────────
TRAIN_DIR = r"D:\My life\EECE\4th year\2nd term\Neural Networks\Project\audio-dataset\Train"   # e.g. "data/train"
TEST_DIR  = r"D:\My life\EECE\4th year\2nd term\Neural Networks\Project\audio-dataset\Test"    # e.g. "data/test"

# Spectrogram settings
SAMPLE_RATE    = 16000   # target sample rate (Hz); audio is resampled if needed
N_MELS         = 128     # number of mel frequency bins
HOP_LENGTH     = 512     # samples between successive frames
N_FFT          = 1024    # FFT window size
IMAGE_SIZE     = 128     # final spectrogram image: IMAGE_SIZE × IMAGE_SIZE pixels
TOP_DB         = 80      # dynamic range for amplitude-to-dB conversion


# ─────────────────────────────────────────────
#  HELPER: discover files
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")

def _discover_files(root_dir: str) -> list[tuple[str, int]]:
    """
    Walk root_dir and return a list of (filepath, label) tuples.

    Expected directory layout (ALL audio files flat in one folder):
        root_dir/
            M20n_6.wav      ← label = 6  (last char before extension)
            M21f_3.wav      ← label = 3
            ...

    The label is extracted from the last character of the filename stem.
    For example:  "M20n_6.wav"  →  stem = "M20n_6"  →  label = int("6") = 6
    Files whose last character is not a digit 0–9 are skipped with a warning.
    """
    samples = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir!r}")

    for fname in sorted(os.listdir(root_dir)):
        if not fname.lower().endswith(SUPPORTED_EXTENSIONS):
            continue  # skip non-audio files

        stem = os.path.splitext(fname)[0]   # e.g. "M20n_6"
        last_char = stem[-1]                # e.g. "6"

        try:
            label = int(last_char)          # must be 0–9
        except ValueError:
            print(f"[dataset] WARNING: skipping '{fname}' — last char '{last_char}' is not a digit")
            continue

        samples.append((os.path.join(root_dir, fname), label))

    if not samples:
        raise RuntimeError(f"No labelled audio files found under {root_dir!r}. "
                           "Check TRAIN_DIR / TEST_DIR and that filenames end with a digit.")

    # Print a quick label distribution so you can spot imbalances early
    from collections import Counter
    dist = Counter(label for _, label in samples)
    dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
    print(f"[dataset] Found {len(samples)} samples in '{root_dir}'  |  {dist_str}")
    return samples


# ─────────────────────────────────────────────
#  HELPER: load waveform
# ─────────────────────────────────────────────
def _load_waveform(
    filepath: str,
    sr: int = SAMPLE_RATE,
    # ── hooks for audio augmentation (Part b) ─────────────────────────
    speed_factor: float | None = None,   # e.g. 1.03 = speed up 3 %
    noise_std: float | None = None,      # e.g. 0.005 = light noise
) -> np.ndarray:
    """
    Load an audio file and return a mono waveform resampled to `sr` Hz.

    Audio augmentation parameters (for Part b):
        speed_factor : float or None
            If given, the waveform is time-stretched by this factor.
            Values > 1 speed up; values < 1 slow down.
        noise_std : float or None
            If given, additive white Gaussian noise with this std-dev
            is mixed into the waveform.
    """
    y, orig_sr = librosa.load(filepath, sr=sr, mono=True)

    # ── Part (b) audio augmentation hooks ──────────────────────────────
    if speed_factor is not None:
        # time_stretch changes duration without affecting pitch
        y = librosa.effects.time_stretch(y, rate=speed_factor)

    if noise_std is not None:
        noise = np.random.randn(len(y)).astype(np.float32) * noise_std
        y = y + noise

    return y


# ─────────────────────────────────────────────
#  HELPER: waveform → log-mel spectrogram image
# ─────────────────────────────────────────────
def _to_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    image_size: int = IMAGE_SIZE,
    # ── hooks for image augmentation (Part c) ─────────────────────────
    squeeze_factor: float | None = None,   # e.g. 0.97 = squeeze 3 %
    expand_factor: float | None = None,    # e.g. 1.03 = expand 3 %
    image_noise_std: float | None = None,  # e.g. 0.02 = 2 % pixel noise
) -> np.ndarray:
    """
    Convert a mono waveform to a fixed-size log-mel spectrogram.

    Pipeline:
        1. Compute mel spectrogram (power).
        2. Convert power → dB (log scale), clip to TOP_DB dynamic range.
        3. Normalise dB values to [0, 1].
        4. Resize to (image_size × image_size) pixels using bilinear interpolation.

    Image augmentation parameters (for Part c):
        squeeze_factor : float or None
            Horizontally squeeze by this factor (< 1).
        expand_factor : float or None
            Horizontally expand by this factor (> 1).
        image_noise_std : float or None
            Add Gaussian noise to the normalised spectrogram pixels.

    Returns
    -------
    np.ndarray of shape (image_size, image_size), dtype float32, range [0, 1].
    """
    # Step 1 – mel spectrogram (power)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )

    # Step 2 – convert to dB, then normalise to [0, 1]
    mel_db = librosa.power_to_db(mel, top_db=TOP_DB)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Step 3 – resize to a fixed square using PIL (bilinear)
    img = Image.fromarray((mel_db * 255).astype(np.uint8), mode="L")

    # ── Part (c) image augmentation hooks ──────────────────────────────
    w, h = img.size                # PIL: (width, height)
    if squeeze_factor is not None:
        new_w = max(1, int(w * squeeze_factor))
        img = img.resize((new_w, h), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if expand_factor is not None:
        new_w = int(w * expand_factor)
        img = img.resize((new_w, h), Image.BILINEAR)
        # crop centre back to original width
        left = (new_w - w) // 2
        img = img.crop((left, 0, left + w, h))

    img = img.resize((image_size, image_size), Image.BILINEAR)

    # Back to float32 [0, 1]
    out = np.array(img, dtype=np.float32) / 255.0

    # ── Part (c) image noise hook ───────────────────────────────────────
    if image_noise_std is not None:
        noise = np.random.randn(*out.shape).astype(np.float32) * image_noise_std
        out = np.clip(out + noise, 0.0, 1.0)

    return out   # shape: (image_size, image_size)


# ─────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────
class SpeechDataset(Dataset):
    """
    PyTorch Dataset for the Arabic digit speech recognition task.

    Each item is a tuple (spectrogram_tensor, label_tensor) where:
        spectrogram_tensor : float32 tensor, shape (1, IMAGE_SIZE, IMAGE_SIZE)
            Single-channel grayscale spectrogram, values in [0, 1].
        label_tensor : long tensor, scalar
            Integer class label (0–9).

    Parameters
    ----------
    root_dir : str
        Path to a directory with sub-folders named 0–9.
    transform : torchvision.transforms or None
        Optional extra transforms applied to the PIL image (for augmentation).
    audio_augment_kwargs : dict or None
        Keyword arguments forwarded to _load_waveform() for audio augmentation
        (Part b).  Example: {"speed_factor": 1.03, "noise_std": 0.005}
    image_augment_kwargs : dict or None
        Keyword arguments forwarded to _to_spectrogram() for image augmentation
        (Part c).  Example: {"squeeze_factor": 0.97}
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        audio_augment_kwargs: dict | None = None,
        image_augment_kwargs: dict | None = None,
    ):
        self.samples               = _discover_files(root_dir)
        self.transform             = transform
        self.audio_augment_kwargs  = audio_augment_kwargs  or {}
        self.image_augment_kwargs  = image_augment_kwargs  or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, label = self.samples[idx]

        # 1. Load waveform (with optional audio augmentation)
        y = _load_waveform(filepath, **self.audio_augment_kwargs)

        # 2. Convert to log-mel spectrogram (with optional image augmentation)
        spec = _to_spectrogram(y, **self.image_augment_kwargs)
        # spec shape: (IMAGE_SIZE, IMAGE_SIZE), float32, [0, 1]

        # 3. Add channel dimension → (1, IMAGE_SIZE, IMAGE_SIZE)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        # 4. Apply any additional torchvision transforms
        if self.transform is not None:
            spec_tensor = self.transform(spec_tensor)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return spec_tensor, label_tensor


# ─────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────
def get_datasets(
    train_dir: str = TRAIN_DIR,
    test_dir:  str = TEST_DIR,
    audio_augment_kwargs: dict | None = None,
    image_augment_kwargs: dict | None = None,
) -> tuple[SpeechDataset, SpeechDataset]:
    """
    Build and return (train_dataset, test_dataset).

    The test dataset never receives augmentation, regardless of the kwargs passed.
    """
    train_dataset = SpeechDataset(
        root_dir=train_dir,
        audio_augment_kwargs=audio_augment_kwargs,
        image_augment_kwargs=image_augment_kwargs,
    )
    test_dataset = SpeechDataset(
        root_dir=test_dir,
        # no augmentation on test data
    )
    return train_dataset, test_dataset


# ─────────────────────────────────────────────
#  Quick sanity check  (run this file directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[dataset] Current working directory: {os.getcwd()}")
    print(f"[dataset] Script path: {os.path.abspath(__file__)}")
    print(f"[dataset] TRAIN_DIR exists: {os.path.isdir(TRAIN_DIR)}  ->  {TRAIN_DIR}")
    print(f"[dataset] TEST_DIR exists: {os.path.isdir(TEST_DIR)}  ->  {TEST_DIR}")

    train_ds, test_ds = get_datasets()
    x, y = train_ds[0]
    print(f"Sample 0 → spec shape: {x.shape}  label: {y.item()}")
    print(f"Train size: {len(train_ds)}   Test size: {len(test_ds)}")
    print(f"Spec min/max: {x.min():.4f} / {x.max():.4f}")