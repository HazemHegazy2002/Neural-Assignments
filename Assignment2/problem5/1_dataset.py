"""
dataset.py
----------
Handles all data preparation for Problem 5 - ReducedMNIST experiments.

Data source: Kaggle ReducedMNIST dataset (already downloaded).

Responsibilities:
    1. Read images from the Kaggle folder structure.
    2. Create THREE fixed training splits:
           - 350 real samples per digit  (3,500 total)
           - 750 real samples per digit  (7,500 total)
           - 1000 real samples per digit (10,000 total)
    3. Use ALL available test images (up to 200 per digit).
    4. Save all splits to disk as .pt files so every experiment
       uses EXACTLY the same samples (reproducibility).
    5. Provide a PyTorch Dataset class for loading any split.
"""

import os
import sys


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

import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import Counter


# -------------------------------------------------
#  CONFIG  -- SET YOUR PATH HERE
# -------------------------------------------------
ROOT_DIR   = r"D:\My life\EECE\4th year\2nd term\Neural Networks\Project\Reduced MNIST Data"
TRAIN_DIR  = os.path.join(ROOT_DIR, "train")
TEST_DIR   = os.path.join(ROOT_DIR, "test")

SPLITS_DIR = "data/splits"   # where .pt files will be saved

SAMPLES_PER_DIGIT_TRAIN = [350, 750, 1000]   # three training split sizes
SAMPLES_PER_DIGIT_TEST  = 200                 # max test samples per digit
NUM_CLASSES             = 10                  # digits 0-9
SEED                    = 42

# Image transform: grayscale PIL -> float32 tensor [0,1] shape (1,28,28)
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),               # -> float32 [0,1]
])

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp")


# -------------------------------------------------
#  STEP 1 - Scan folder and collect (filepath, label)
# -------------------------------------------------
def _scan_folder(root):
    """
    Walk root/0/, root/1/, ..., root/9/ and collect
    a dict: { digit(int) -> [filepath, ...] }
    """
    samples = {}
    for digit in range(NUM_CLASSES):
        digit_dir = os.path.join(root, str(digit))
        if not os.path.isdir(digit_dir):
            raise FileNotFoundError(
                f"Expected folder not found: {digit_dir}\n"
                f"Make sure ROOT_DIR is set correctly in dataset.py"
            )
        files = [
            os.path.join(digit_dir, f)
            for f in sorted(os.listdir(digit_dir))
            if f.lower().endswith(SUPPORTED_EXT)
        ]
        samples[digit] = files
        print(f"  Digit {digit}: {len(files)} images found")

    return samples


# -------------------------------------------------
#  STEP 2 - Load images into tensors
# -------------------------------------------------
def _load_images(file_list):
    """
    Load a list of image file paths and return a stacked tensor.
    Returns: float32 tensor shape (N, 1, 28, 28)
    """
    tensors = []
    for path in file_list:
        img = Image.open(path).convert("L")   # force grayscale
        tensors.append(TRANSFORM(img))
    return torch.stack(tensors)               # (N, 1, 28, 28)


# -------------------------------------------------
#  STEP 3 - Build a split dict from sampled files
# -------------------------------------------------
def _build_split(samples_dict, n_per_digit, rng):
    """
    Randomly sample n_per_digit files from each class,
    load them, and return {images: tensor, labels: tensor}.
    """
    all_images = []
    all_labels = []

    for digit in range(NUM_CLASSES):
        files = samples_dict[digit]
        if len(files) < n_per_digit:
            raise ValueError(
                f"Digit {digit} has only {len(files)} images "
                f"but {n_per_digit} were requested."
            )
        chosen = rng.sample(files, n_per_digit)
        imgs   = _load_images(chosen)         # (n_per_digit, 1, 28, 28)
        all_images.append(imgs)
        all_labels.extend([digit] * n_per_digit)

    images = torch.cat(all_images, dim=0)                   # (N, 1, 28, 28)
    labels = torch.tensor(all_labels, dtype=torch.long)     # (N,)
    return {"images": images, "labels": labels}


# -------------------------------------------------
#  MAIN: create and save all splits
# -------------------------------------------------
def prepare_splits(force=False):
    """
    Create and save all fixed data splits from the Kaggle folder.

    Parameters
    ----------
    force : bool
        If True, recreate splits even if they already exist.
        If False (default), skip if all files already present.

    Files saved:
        data/splits/train_350.pt
        data/splits/train_750.pt
        data/splits/train_1000.pt
        data/splits/test_200.pt
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Check if all splits already exist
    expected = [f"train_{n}.pt" for n in SAMPLES_PER_DIGIT_TRAIN]
    expected.append("test_200.pt")
    all_exist = all(
        os.path.exists(os.path.join(SPLITS_DIR, f)) for f in expected
    )

    if all_exist and not force:
        print("[dataset] All splits already exist. Skipping creation.")
        print("[dataset] Delete data/splits/ or use force=True to recreate.")
        _verify_splits()
        return

    # ── Scan train folder ─────────────────────────────────────────────
    print(f"\n[dataset] Scanning TRAIN folder: {TRAIN_DIR}")
    train_samples = _scan_folder(TRAIN_DIR)

    # ── Create training splits ────────────────────────────────────────
    for n in SAMPLES_PER_DIGIT_TRAIN:
        print(f"\n[dataset] Building train_{n}.pt ...")
        rng   = random.Random(SEED + n)      # unique seed per split size
        split = _build_split(train_samples, n, rng)
        path  = os.path.join(SPLITS_DIR, f"train_{n}.pt")
        torch.save(split, path)
        print(f"[dataset] Saved -> {split['images'].shape}")

    # ── Scan test folder ──────────────────────────────────────────────
    print(f"\n[dataset] Scanning TEST folder: {TEST_DIR}")
    test_samples = _scan_folder(TEST_DIR)

    # Use up to SAMPLES_PER_DIGIT_TEST per digit
    print(f"\n[dataset] Building test_200.pt ...")
    test_rng = random.Random(SEED + 9999)
    test_n   = min(
        SAMPLES_PER_DIGIT_TEST,
        min(len(test_samples[d]) for d in range(NUM_CLASSES))
    )
    if test_n < SAMPLES_PER_DIGIT_TEST:
        print(f"[dataset] WARNING: only {test_n} test images per digit available "
              f"(requested {SAMPLES_PER_DIGIT_TEST})")

    test_split = _build_split(test_samples, test_n, test_rng)
    test_path  = os.path.join(SPLITS_DIR, "test_200.pt")
    torch.save(test_split, test_path)
    print(f"[dataset] Saved -> {test_split['images'].shape}")

    print("\n[dataset] All splits saved successfully.")
    _verify_splits()


# -------------------------------------------------
#  VERIFY - sanity check saved splits
# -------------------------------------------------
def _verify_splits():
    """Load each saved split and print a quick summary."""
    print("\n[dataset] Verifying splits ...")

    files = {f"train_{n}": os.path.join(SPLITS_DIR, f"train_{n}.pt")
             for n in SAMPLES_PER_DIGIT_TRAIN}
    files["test_200"] = os.path.join(SPLITS_DIR, "test_200.pt")

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"  {name:12s} | NOT FOUND")
            continue
        split    = torch.load(path, weights_only=True)
        images   = split["images"]
        labels   = split["labels"]
        counts   = [(labels == c).sum().item() for c in range(NUM_CLASSES)]
        balanced = len(set(counts)) == 1
        print(f"  {name:12s} | {str(images.shape):25s} | "
              f"range [{images.min():.2f}, {images.max():.2f}] | "
              f"balanced: {'OK' if balanced else 'FAIL'} "
              f"({counts[0]} per digit)")


# -------------------------------------------------
#  PyTorch Dataset class
# -------------------------------------------------
class ReducedMNIST(Dataset):
    """
    PyTorch Dataset that loads a pre-saved ReducedMNIST split.

    Parameters
    ----------
    split_path : str
        Path to a .pt file created by prepare_splits().
        e.g. "data/splits/train_350.pt"
    extra_images : torch.Tensor or None
        Augmented images shape (M, 1, 28, 28) to concatenate.
    extra_labels : torch.Tensor or None
        Labels for extra images, shape (M,).
    transform : callable or None
        Optional transform applied at __getitem__ time.
    """

    def __init__(self, split_path, extra_images=None,
                 extra_labels=None, transform=None):
        split  = torch.load(split_path, weights_only=True)
        images = split["images"]
        labels = split["labels"]

        if extra_images is not None and extra_labels is not None:
            images = torch.cat([images, extra_images], dim=0)
            labels = torch.cat([labels, extra_labels], dim=0)

        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# -------------------------------------------------
#  Convenience loaders (used by augment.py & train.py)
# -------------------------------------------------
def load_split(n_real):
    """
    Load training split for n_real samples per digit.
    n_real must be 350, 750, or 1000.
    Returns (images, labels) tensors.
    """
    assert n_real in SAMPLES_PER_DIGIT_TRAIN, \
        f"n_real must be one of {SAMPLES_PER_DIGIT_TRAIN}, got {n_real}"
    path  = os.path.join(SPLITS_DIR, f"train_{n_real}.pt")
    split = torch.load(path, weights_only=True)
    return split["images"], split["labels"]


def load_test():
    """Load the fixed test split. Returns (images, labels) tensors."""
    path  = os.path.join(SPLITS_DIR, "test_200.pt")
    split = torch.load(path, weights_only=True)
    return split["images"], split["labels"]


# -------------------------------------------------
#  Run standalone
# -------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  ReducedMNIST - Data Preparation from Kaggle folder")
    print("=" * 60)
    prepare_splits(force=False)
    print("\n[dataset] Done. Ready for augmentation and training.")