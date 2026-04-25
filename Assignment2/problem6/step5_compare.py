import csv
import json
import os
import platform
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# -----------------------------------------
# PATHS
# -----------------------------------------
RESULTS_DIR = "./results"
MODELS_DIR = "./results/models"
os.makedirs(RESULTS_DIR, exist_ok=True)

PART2_JSON = f"{RESULTS_DIR}/problem6_part2_results.json"
PART3_JSON = f"{RESULTS_DIR}/problem6_part3_results.json"
PART3_CSV = f"{RESULTS_DIR}/problem6_part3_results.csv"
PART3_ACC_PLOT = f"{RESULTS_DIR}/problem6_part3_accuracy_comparison.png"
PART3_LOSS_PLOT = f"{RESULTS_DIR}/problem6_part3_loss_comparison.png"

# -----------------------------------------
# HYPERPARAMETERS
# -----------------------------------------
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
NUM_CLASSES = 10
NOISE_DIM = 100
EMBED_DIM = 50
CHANNELS = 1

REAL_BASE_PER_DIGIT = 350
REAL_TARGET_PER_DIGIT = 1000

# Part 3 requires any reasonable mix of augmented and synthetic data.
AUGMENTED_PER_DIGIT = 500
SYNTHETIC_PER_DIGIT = 1000

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------
# DEVICE
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def machine_setup():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }


# -----------------------------------------
# LENET-5
# -----------------------------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


# -----------------------------------------
# GENERATOR (same architecture as step2)
# -----------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM + EMBED_DIM, 7 * 7 * 256),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.ReLU(),
        )
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, CHANNELS, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_vec = self.label_embed(labels)
        x = torch.cat([noise, label_vec], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        return self.conv_layers(x)


def build_real_pools(full_data, full_targets, max_per_digit, seed):
    pools = {}
    generator = torch.Generator().manual_seed(seed)
    for digit in range(NUM_CLASSES):
        idx = (full_targets == digit).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(idx), generator=generator)
        selected = idx[perm[:max_per_digit]]
        pools[digit] = {
            "images": full_data[selected].unsqueeze(1),
            "labels": full_targets[selected],
        }
    return pools


def real_subset_from_pools(real_pools, n_per_digit):
    images = []
    labels = []
    for digit in range(NUM_CLASSES):
        images.append(real_pools[digit]["images"][:n_per_digit])
        labels.append(real_pools[digit]["labels"][:n_per_digit])
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


def generate_synthetic(generator, num_per_digit):
    generator.eval()
    all_images = []
    all_labels = []
    with torch.no_grad():
        for digit in range(NUM_CLASSES):
            noise = torch.randn(num_per_digit, NOISE_DIM, device=device)
            labels = torch.full((num_per_digit,), digit, dtype=torch.long, device=device)
            generated = generator(noise, labels)
            all_images.append(generated.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_images, dim=0), torch.cat(all_labels, dim=0)


def augment_one_image(img):
    """Augment a normalized tensor image in [-1, 1] using rotation/shift/noise."""
    mode = random.choice(["rotate", "shift", "noise", "rotate_shift_noise"])
    out = img.clone()

    if mode in ("rotate", "rotate_shift_noise"):
        angle = random.uniform(-20.0, 20.0)
    else:
        angle = 0.0

    if mode in ("shift", "rotate_shift_noise"):
        tx = random.randint(-3, 3)
        ty = random.randint(-3, 3)
    else:
        tx = 0
        ty = 0

    if angle != 0.0 or tx != 0 or ty != 0:
        out = TF.affine(
            out,
            angle=angle,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=-1.0,
        )

    if mode in ("noise", "rotate_shift_noise"):
        noise_std = random.uniform(0.05, 0.15)
        out = torch.clamp(out + noise_std * torch.randn_like(out), -1.0, 1.0)

    return out


def create_augmented_dataset(real_images, real_labels, n_per_digit):
    aug_images = []
    aug_labels = []

    for digit in range(NUM_CLASSES):
        idx = (real_labels == digit).nonzero(as_tuple=True)[0]
        for _ in range(n_per_digit):
            src_i = idx[random.randrange(len(idx))]
            src = real_images[src_i]
            aug = augment_one_image(src)
            aug_images.append(aug)
            aug_labels.append(digit)

    return torch.stack(aug_images, dim=0), torch.tensor(aug_labels, dtype=torch.long)


def train_lenet(train_images, train_labels, test_images, test_labels, epochs, description):
    print(f"\n{'=' * 76}")
    print(f"TRAINING: {description}")
    print(f"Train size: {len(train_images)} | Test size: {len(test_images)}")
    print(f"{'=' * 76}")

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LeNet5(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    losses = []
    train_start = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1:>2}/{epochs}] Loss: {avg_loss:.4f}")

    train_time_ms = (time.time() - train_start) * 1000.0

    test_start = time.time()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_imgs)
            predicted = torch.argmax(outputs, dim=1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    test_time_ms = (time.time() - test_start) * 1000.0
    accuracy = 100.0 * correct / total
    print(f"Accuracy: {accuracy:.2f}% | Train: {train_time_ms:.1f} ms | Test: {test_time_ms:.1f} ms")
    return accuracy, train_time_ms, test_time_ms, losses


set_seed(SEED)
setup = machine_setup()

print("\nMachine setup:")
for k, v in setup.items():
    print(f"  {k}: {v}")

if not os.path.exists(PART2_JSON):
    raise FileNotFoundError(
        f"Missing {PART2_JSON}. Run step4_train_lenet.py first to generate Part 2 results."
    )

with open(PART2_JSON, "r", encoding="utf-8") as f:
    part2_rows = json.load(f)

part2_350 = [row for row in part2_rows if row["real_per_digit"] == REAL_BASE_PER_DIGIT]
if not part2_350:
    raise RuntimeError("Part 2 results do not contain real_per_digit == 350 rows.")

best_part2 = max(part2_350, key=lambda row: row["accuracy"])

print("\nBest Part 2 (350-real runs):")
print(
    f"  real/digit={best_part2['real_per_digit']} "
    f"generated/digit={best_part2['generated_per_digit']} "
    f"accuracy={best_part2['accuracy']:.2f}%"
)

# Load test set from step1 (200 per digit)
test_images = torch.load(f"{RESULTS_DIR}/test_images.pt")
test_labels = torch.load(f"{RESULTS_DIR}/test_labels.pt")

# Build real pools for 350 and 1000 real per digit
full_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
full_data = full_train.data.float() / 127.5 - 1.0
full_targets = full_train.targets
real_pools = build_real_pools(full_data, full_targets, REAL_TARGET_PER_DIGIT, SEED)

real_350_images, real_350_labels = real_subset_from_pools(real_pools, REAL_BASE_PER_DIGIT)
real_1000_images, real_1000_labels = real_subset_from_pools(real_pools, REAL_TARGET_PER_DIGIT)

# Load generator
generator_path = f"{MODELS_DIR}/generator.pth"
if not os.path.exists(generator_path):
    raise FileNotFoundError(
        f"Missing generator model: {generator_path}. Run step2_train_cgan.py first."
    )

generator = Generator().to(device)
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.eval()

# Part 3 chosen combination: 350 real + augmented + synthetic
print("\nCreating augmented + synthetic training set...")
aug_images, aug_labels = create_augmented_dataset(
    real_350_images,
    real_350_labels,
    AUGMENTED_PER_DIGIT,
)
syn_images, syn_labels = generate_synthetic(generator, SYNTHETIC_PER_DIGIT)

combo_images = torch.cat([real_350_images, aug_images, syn_images], dim=0)
combo_labels = torch.cat([real_350_labels, aug_labels, syn_labels], dim=0)

combo_desc = (
    f"350 real/digit + {AUGMENTED_PER_DIGIT} augmented/digit "
    f"+ {SYNTHETIC_PER_DIGIT} synthetic/digit"
)

combo_acc, combo_train_ms, combo_test_ms, combo_losses = train_lenet(
    combo_images,
    combo_labels,
    test_images,
    test_labels,
    EPOCHS,
    combo_desc,
)

real1000_acc, real1000_train_ms, real1000_test_ms, real1000_losses = train_lenet(
    real_1000_images,
    real_1000_labels,
    test_images,
    test_labels,
    EPOCHS,
    "1000 real per digit (reference)",
)

# Save and plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(combo_losses, label="350 real + aug + syn", color="blue")
plt.plot(real1000_losses, label="1000 real only", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Problem 6 Part 3: Training Loss Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PART3_LOSS_PLOT, dpi=150)
plt.close()

comparison_rows = [
    {
        "experiment": "Best Part 2 (350-real runs)",
        "real_per_digit": int(best_part2["real_per_digit"]),
        "augmented_per_digit": 0,
        "synthetic_per_digit": int(best_part2["generated_per_digit"]),
        "accuracy": float(best_part2["accuracy"]),
        "train_time_ms": float(best_part2["train_time_ms"]),
        "test_time_ms": float(best_part2["test_time_ms"]),
    },
    {
        "experiment": "Chosen Part 3 combo (350 real + augmented + synthetic)",
        "real_per_digit": REAL_BASE_PER_DIGIT,
        "augmented_per_digit": AUGMENTED_PER_DIGIT,
        "synthetic_per_digit": SYNTHETIC_PER_DIGIT,
        "accuracy": float(combo_acc),
        "train_time_ms": float(combo_train_ms),
        "test_time_ms": float(combo_test_ms),
    },
    {
        "experiment": "1000 real only",
        "real_per_digit": REAL_TARGET_PER_DIGIT,
        "augmented_per_digit": 0,
        "synthetic_per_digit": 0,
        "accuracy": float(real1000_acc),
        "train_time_ms": float(real1000_train_ms),
        "test_time_ms": float(real1000_test_ms),
    },
]

with open(PART3_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "machine_setup": setup,
            "comparison": comparison_rows,
            "note": "Part 3 compares best Part 2 run (350-real only setup) vs chosen augmented+synthetic combo vs 1000-real reference.",
        },
        f,
        indent=2,
    )

with open(PART3_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "experiment",
            "real_per_digit",
            "augmented_per_digit",
            "synthetic_per_digit",
            "accuracy",
            "train_time_ms",
            "test_time_ms",
        ],
    )
    writer.writeheader()
    writer.writerows(comparison_rows)

# Plot accuracy bars
plt.figure(figsize=(11, 6))
labels = [
    "Best Part2\n(350-real)",
    "350 real +\naug + syn",
    "1000 real\nonly",
]
acc_values = [
    comparison_rows[0]["accuracy"],
    comparison_rows[1]["accuracy"],
    comparison_rows[2]["accuracy"],
]
bars = plt.bar(labels, acc_values, color=["gray", "blue", "red"], alpha=0.8)
for bar, v in zip(bars, acc_values):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f}%", ha="center")
plt.ylabel("Accuracy (%)")
plt.title("Problem 6 Part 3: Accuracy Comparison")
plt.ylim(min(acc_values) - 3.0, 100.0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PART3_ACC_PLOT, dpi=150)
plt.close()

print("\n" + "=" * 100)
print("PROBLEM 6 PART 3 COMPARISON")
print("=" * 100)
print(
    f"{'Experiment':<52} {'Real':>6} {'Aug':>6} {'Synth':>8} {'Acc':>9} {'Train ms':>12} {'Test ms':>10}"
)
print("-" * 100)
for row in comparison_rows:
    print(
        f"{row['experiment']:<52} "
        f"{row['real_per_digit']:>6} "
        f"{row['augmented_per_digit']:>6} "
        f"{row['synthetic_per_digit']:>8} "
        f"{row['accuracy']:>8.2f}% "
        f"{row['train_time_ms']:>12.1f} "
        f"{row['test_time_ms']:>10.1f}"
    )
print("-" * 100)

gain_vs_1000 = combo_acc - real1000_acc
real_data_reduction_pct = 100.0 * (1.0 - REAL_BASE_PER_DIGIT / REAL_TARGET_PER_DIGIT)

print(f"\nChosen combo vs 1000-real accuracy difference: {gain_vs_1000:+.2f}%")
print(
    f"Real data used in chosen combo: {REAL_BASE_PER_DIGIT}/digit instead of {REAL_TARGET_PER_DIGIT}/digit "
    f"({real_data_reduction_pct:.0f}% fewer real samples)."
)

print(f"\nSaved JSON: {PART3_JSON}")
print(f"Saved CSV : {PART3_CSV}")
print(f"Saved plot: {PART3_ACC_PLOT}")
print(f"Saved plot: {PART3_LOSS_PLOT}")
print("\nStep 5 complete.")