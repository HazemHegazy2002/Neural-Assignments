import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
SAMPLES_PER_DIGIT_TRAIN = 1000
SAMPLES_PER_DIGIT_TEST  = 200
NUM_CLASSES             = 10
SEED                    = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─────────────────────────────────────────
# DOWNLOAD MNIST
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train = torchvision.datasets.MNIST(
    root='./data', train=True,
    download=True, transform=transform
)
full_test = torchvision.datasets.MNIST(
    root='./data', train=False,
    download=True, transform=transform
)

# ─────────────────────────────────────────
# REDUCE DATASET
# ─────────────────────────────────────────
def get_reduced_indices(dataset, samples_per_digit, seed=42):
    rng = np.random.RandomState(seed)
    indices = []
    targets = np.array(dataset.targets)
    for digit in range(10):
        digit_idx = np.where(targets == digit)[0]
        selected  = rng.choice(digit_idx, samples_per_digit, replace=False)
        indices.extend(selected.tolist())
    return indices

train_indices = get_reduced_indices(full_train, SAMPLES_PER_DIGIT_TRAIN)
test_indices  = get_reduced_indices(full_test,  SAMPLES_PER_DIGIT_TEST)

train_dataset = Subset(full_train, train_indices)
test_dataset  = Subset(full_test,  test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f"Train size : {len(train_dataset)}")
print(f"Test size  : {len(test_dataset)}")
print("✅ Step 1 Complete — Data ready!")