import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
DATA_DIR    = "./data"
RESULTS_DIR = "./results"
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
SAMPLES_PER_DIGIT_TRAIN = 350   # Problem 6 requirement
SAMPLES_PER_DIGIT_TEST  = 200   # ReducedMNIST requirement
NUM_DIGITS              = 10    # digits 0-9

# ─────────────────────────────────────────
# DOWNLOAD FULL MNIST
# ─────────────────────────────────────────
print("Downloading MNIST dataset...")

# Transform: convert to tensor and normalize to [-1, 1]
# We use [-1, 1] because GAN output uses Tanh activation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download train set (60,000 images)
full_train = torchvision.datasets.MNIST(
    root      = DATA_DIR,
    train     = True,
    download  = True,
    transform = transform
)

# Download test set (10,000 images)
full_test = torchvision.datasets.MNIST(
    root      = DATA_DIR,
    train     = False,
    download  = True,
    transform = transform
)

print(f"Full MNIST train size : {len(full_train)}")
print(f"Full MNIST test size  : {len(full_test)}")

# ─────────────────────────────────────────
# FUNCTION: Reduce dataset
# ─────────────────────────────────────────
def reduce_dataset(dataset, samples_per_digit):
    """
    Randomly select samples_per_digit from each digit
    Returns:
        images : tensor [total, 1, 28, 28]
        labels : tensor [total]
    """
    all_images = []
    all_labels = []

    # Get all images and labels
    all_data   = dataset.data.float() / 127.5 - 1.0
    # Normalize to [-1, 1] same as transform
    all_targets = dataset.targets

    print(f"\nSelecting {samples_per_digit} samples per digit:")

    for digit in range(NUM_DIGITS):
        # Find all indices for this digit
        digit_indices = (all_targets == digit).nonzero(as_tuple=True)[0]

        # Randomly select samples_per_digit indices
        selected = torch.randperm(len(digit_indices))[:samples_per_digit]
        selected_indices = digit_indices[selected]

        # Get selected images and labels
        digit_images = all_data[selected_indices]
        digit_labels = all_targets[selected_indices]

        all_images.append(digit_images)
        all_labels.append(digit_labels)

        print(f"  Digit {digit}: {len(digit_images)} images selected")

    # Stack all digits together
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Add channel dimension [N, 28, 28] → [N, 1, 28, 28]
    all_images = all_images.unsqueeze(1)

    return all_images, all_labels

# ─────────────────────────────────────────
# CREATE REDUCED DATASETS
# ─────────────────────────────────────────
print("\n" + "="*50)
print("CREATING REDUCED TRAIN SET (350 per digit)")
print("="*50)
train_images, train_labels = reduce_dataset(full_train, SAMPLES_PER_DIGIT_TRAIN)

print("\n" + "="*50)
print("CREATING REDUCED TEST SET (200 per digit)")
print("="*50)
test_images, test_labels = reduce_dataset(full_test, SAMPLES_PER_DIGIT_TEST)

# ─────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────
print("\n" + "="*50)
print("DATASET SUMMARY")
print("="*50)
print(f"Train images shape : {train_images.shape}")
print(f"Train labels shape : {train_labels.shape}")
print(f"Test images shape  : {test_images.shape}")
print(f"Test labels shape  : {test_labels.shape}")
print(f"Image value range  : [{train_images.min():.1f}, {train_images.max():.1f}]")
print(f"Total train images : {len(train_images)}")
print(f"Total test images  : {len(test_images)}")

# ─────────────────────────────────────────
# VISUALIZE SAMPLES
# ─────────────────────────────────────────
print("\nVisualizing sample images...")

fig, axes = plt.subplots(2, 10, figsize=(20, 4))

for digit in range(10):
    # Get first image of each digit from train
    digit_idx = (train_labels == digit).nonzero(as_tuple=True)[0][0]
    img = train_images[digit_idx].squeeze()

    # Top row: train samples
    axes[0, digit].imshow(img, cmap='gray')
    axes[0, digit].set_title(f'Digit {digit}')
    axes[0, digit].axis('off')

    # Bottom row: another sample
    digit_idx2 = (train_labels == digit).nonzero(as_tuple=True)[0][1]
    img2 = train_images[digit_idx2].squeeze()
    axes[1, digit].imshow(img2, cmap='gray')
    axes[1, digit].axis('off')

fig.suptitle('ReducedMNIST Samples (350 per digit)', fontsize=14)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/sample_images.png")
plt.show()
print("Sample images saved!")

# ─────────────────────────────────────────
# SAVE DATASETS
# ─────────────────────────────────────────
print("\nSaving datasets...")

torch.save(train_images, f"{RESULTS_DIR}/train_images.pt")
torch.save(train_labels, f"{RESULTS_DIR}/train_labels.pt")
torch.save(test_images,  f"{RESULTS_DIR}/test_images.pt")
torch.save(test_labels,  f"{RESULTS_DIR}/test_labels.pt")

print(f"Saved:")
print(f"  train_images.pt → {train_images.shape}")
print(f"  train_labels.pt → {train_labels.shape}")
print(f"  test_images.pt  → {test_images.shape}")
print(f"  test_labels.pt  → {test_labels.shape}")

print("\n✅ Step 1 Complete — ReducedMNIST ready!")