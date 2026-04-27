"""
augment.py
----------
Generates augmented training samples for Problem 5.

The assignment requires augmentation methods including:
    - Rotation  : rotate digit left or right by a random angle
    - Translation: shift digit randomly in x and y directions
    - Noise     : add a percentage of white (Gaussian) noise

How augmentation works here:
    Given N real images per digit, we need to generate M augmented
    images per digit. We cycle through the real images repeatedly,
    applying a DIFFERENT random transform each time, until we have
    exactly M generated samples per digit.

    Example: 350 real, need 1000 generated
        -> each real image is reused ~3 times with different transforms
        -> result: 1000 new images that look like valid digit variations

Why augmentation helps:
    The model never sees the same exact pixel pattern twice, so it
    learns the SHAPE of digits rather than memorizing specific images.
    This improves generalization to the test set.

Why it can also hurt (important for report):
    If augmentation is too aggressive (e.g. rotating 90 degrees),
    a "6" can look like a "9" and confuse the model. Mild transforms
    work best for digit recognition.
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
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF

from p5_dataset import load_split, NUM_CLASSES, SAMPLES_PER_DIGIT_TRAIN


# -------------------------------------------------
#  AUGMENTATION CONFIG  -- tweak these freely
# -------------------------------------------------
MAX_ROTATION_DEG  = 15      # rotate up to ±15 degrees
MAX_TRANSLATE_PX  = 3       # shift up to ±3 pixels in x and y
NOISE_STD         = 0.05    # Gaussian noise std dev (5% of pixel range)

# Probability that each transform is applied to a given sample
# All three can apply simultaneously to one image
P_ROTATION    = 0.8
P_TRANSLATION = 0.8
P_NOISE       = 0.5


# -------------------------------------------------
#  SINGLE IMAGE AUGMENTATION
# -------------------------------------------------
def augment_one(img: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """
    Apply a random combination of transforms to a single image.

    Parameters
    ----------
    img : torch.Tensor  shape (1, 28, 28), float32, range [0, 1]
    rng : random.Random instance for reproducible randomness

    Returns
    -------
    torch.Tensor  shape (1, 28, 28), float32, range [0, 1]
    """
    # Work on a copy so we never modify the original
    out = img.clone()

    # 1. Random Rotation
    #    Rotates the digit by a small angle. Mimics natural hand tilt.
    #    Fill value = 0 (black background, same as MNIST).
    if rng.random() < P_ROTATION:
        angle = rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        out   = TF.rotate(out, angle=angle, fill=0)

    # 2. Random Translation (affine shift)
    #    Shifts the digit slightly — mimics writing position variation.
    if rng.random() < P_TRANSLATION:
        dx = rng.randint(-MAX_TRANSLATE_PX, MAX_TRANSLATE_PX)
        dy = rng.randint(-MAX_TRANSLATE_PX, MAX_TRANSLATE_PX)
        out = TF.affine(out, angle=0, translate=[dx, dy],
                        scale=1.0, shear=0, fill=0)

    # 3. Gaussian Noise
    #    Adds small pixel noise. Makes model robust to image artifacts.
    if rng.random() < P_NOISE:
        noise = torch.randn_like(out) * NOISE_STD
        out   = torch.clamp(out + noise, 0.0, 1.0)

    return out


# -------------------------------------------------
#  GENERATE N AUGMENTED SAMPLES PER DIGIT
# -------------------------------------------------
def generate_augmented(
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    n_generate: int,
    seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate n_generate augmented samples PER DIGIT from real images.

    Parameters
    ----------
    real_images : torch.Tensor  shape (N, 1, 28, 28)
        The real training images (all digits mixed).
    real_labels : torch.Tensor  shape (N,)
        Labels for real_images.
    n_generate : int
        Number of augmented samples to generate per digit class.
        e.g. 1000 means 1000 new images for digit 0, 1000 for digit 1, etc.
        Total output = n_generate * NUM_CLASSES samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    aug_images : torch.Tensor  shape (n_generate * 10, 1, 28, 28)
    aug_labels : torch.Tensor  shape (n_generate * 10,)
    """
    rng = random.Random(seed)

    aug_images_list = []
    aug_labels_list = []

    for digit in range(NUM_CLASSES):
        # Get all real images for this digit
        mask        = (real_labels == digit)
        digit_imgs  = real_images[mask]       # (n_real_for_digit, 1, 28, 28)
        n_real      = len(digit_imgs)

        if n_real == 0:
            raise ValueError(f"No real images found for digit {digit}!")

        # Generate n_generate augmented samples by cycling through real images
        generated = []
        for i in range(n_generate):
            # Pick source image: cycle through real images repeatedly
            src_idx = i % n_real
            src_img = digit_imgs[src_idx]

            # Apply random augmentation
            aug_img = augment_one(src_img, rng)
            generated.append(aug_img)

        digit_aug = torch.stack(generated)   # (n_generate, 1, 28, 28)
        aug_images_list.append(digit_aug)
        aug_labels_list.extend([digit] * n_generate)

    aug_images = torch.cat(aug_images_list, dim=0)
    aug_labels = torch.tensor(aug_labels_list, dtype=torch.long)

    return aug_images, aug_labels


# -------------------------------------------------
#  CONVENIENCE FUNCTION  (used by experiments.py)
# -------------------------------------------------
def get_augmented_data(n_real: int, n_generate: int):
    """
    Load real split and generate augmented samples in one call.

    Parameters
    ----------
    n_real     : int  -- 350, 750, or 1000 (real samples per digit)
    n_generate : int  -- 0, 1000, 1500, or 2000 (augmented per digit)
                         If 0, returns None, None (no augmentation).

    Returns
    -------
    aug_images : torch.Tensor or None   shape (n_generate*10, 1, 28, 28)
    aug_labels : torch.Tensor or None   shape (n_generate*10,)
    """
    if n_generate == 0:
        return None, None

    real_images, real_labels = load_split(n_real)

    print(f"[augment] Generating {n_generate} samples/digit "
          f"from {n_real} real samples/digit ...")

    aug_images, aug_labels = generate_augmented(
        real_images, real_labels,
        n_generate=n_generate,
        seed=42 + n_real + n_generate    # unique seed per experiment
    )

    print(f"[augment] Done -> aug_images: {aug_images.shape}  "
          f"aug_labels: {aug_labels.shape}")

    return aug_images, aug_labels


# -------------------------------------------------
#  Sanity check / visualize (run standalone)
# -------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")    # no display needed
    import matplotlib.pyplot as plt

    print("=" * 55)
    print("  Augmentation Sanity Check")
    print("=" * 55)

    # Load 350 real samples
    real_images, real_labels = load_split(350)
    print(f"Loaded real images: {real_images.shape}")

    # Generate 1000 augmented per digit
    aug_imgs, aug_lbls = generate_augmented(
        real_images, real_labels, n_generate=1000, seed=42
    )
    print(f"Augmented images : {aug_imgs.shape}")
    print(f"Augmented labels : {aug_lbls.shape}")

    # Verify balance
    for d in range(NUM_CLASSES):
        count = (aug_lbls == d).sum().item()
        print(f"  Digit {d}: {count} augmented samples")

    # Save a visual comparison: original vs augmented (first 5 of digit 3)
    digit_to_show = 3
    mask_real = (real_labels == digit_to_show).nonzero(as_tuple=True)[0][:5]
    mask_aug  = (aug_lbls    == digit_to_show).nonzero(as_tuple=True)[0][:5]

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle(f"Digit {digit_to_show}: Top row = Real, Bottom row = Augmented",
                 fontsize=12)

    for col in range(5):
        axes[0, col].imshow(real_images[mask_real[col]].squeeze(), cmap="gray")
        axes[0, col].axis("off")
        axes[0, col].set_title("Real")

        axes[1, col].imshow(aug_imgs[mask_aug[col]].squeeze(), cmap="gray")
        axes[1, col].axis("off")
        axes[1, col].set_title("Aug")

    plt.tight_layout()
    plt.savefig("augmentation_preview.png", dpi=120)
    print("\n[augment] Preview saved to augmentation_preview.png")
    print("[augment] Done.")