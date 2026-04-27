import csv
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------
# PATHS
# -----------------------------------------
RESULTS_DIR = "./results"
MODELS_DIR = "./results/models"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------
# HYPERPARAMETERS
# -----------------------------------------
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
NUM_CLASSES = 10
NOISE_DIM = 100
EMBED_DIM = 50
CHANNELS = 1

# Problem 6, Part 2 (from PDF table):
# Evaluate 350/750/1000 real samples per digit, with generated counts 0/1000/1500/2000.
REAL_COUNTS = [350, 750, 1000]
GENERATED_COUNTS = [0, 1000, 1500, 2000]

SEED = 42

PART2_JSON = f"{RESULTS_DIR}/problem6_part2_results.json"
PART2_CSV = f"{RESULTS_DIR}/problem6_part2_results.csv"
PART2_HEATMAP = f"{RESULTS_DIR}/problem6_part2_accuracy_heatmap.png"


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
    """Create per-digit real pools once; supports up to 1000 real per digit."""
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


def train_lenet(train_images, train_labels, test_images, test_labels, epochs, description):
    print(f"\n{'=' * 70}")
    print(f"TRAINING: {description}")
    print(f"Train size: {len(train_images)} | Test size: {len(test_images)}")
    print(f"{'=' * 70}")

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    loader_kwargs = {
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    model = LeNet5(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch [{epoch + 1:>2}/{epochs}] Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")
        scheduler.step()

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
    return accuracy, train_time_ms, test_time_ms


def save_results(results):
    with open(PART2_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(PART2_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "real_per_digit",
                "generated_per_digit",
                "train_size_total",
                "test_size_total",
                "accuracy",
                "train_time_ms",
                "test_time_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def plot_heatmap(results):
    matrix = np.zeros((len(GENERATED_COUNTS), len(REAL_COUNTS)), dtype=np.float32)
    for row in results:
        i = GENERATED_COUNTS.index(row["generated_per_digit"])
        j = REAL_COUNTS.index(row["real_per_digit"])
        matrix[i, j] = row["accuracy"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(REAL_COUNTS)))
    ax.set_yticks(range(len(GENERATED_COUNTS)))
    ax.set_xticklabels([str(v) for v in REAL_COUNTS])
    ax.set_yticklabels([str(v) for v in GENERATED_COUNTS])
    ax.set_xlabel("Real samples per digit")
    ax.set_ylabel("Generated samples per digit")
    ax.set_title("Problem 6 Part 2: Accuracy Table")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(PART2_HEATMAP, dpi=150)
    plt.close()


set_seed(SEED)

# -----------------------------------------
# LOAD TEST DATA (200 per digit from step1)
# -----------------------------------------
print("\nLoading ReducedMNIST test set (200 per digit)...")
test_images = torch.load(f"{RESULTS_DIR}/test_images.pt")
test_labels = torch.load(f"{RESULTS_DIR}/test_labels.pt")
print(f"Test shape: {tuple(test_images.shape)}")

# -----------------------------------------
# LOAD FULL MNIST TRAIN AND BUILD REAL POOLS
# -----------------------------------------
print("\nLoading full MNIST train set and selecting up to 1000 real samples per digit...")
full_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
full_data = full_train.data.float() / 127.5 - 1.0
full_targets = full_train.targets
real_pools = build_real_pools(full_data, full_targets, max(REAL_COUNTS), SEED)

# -----------------------------------------
# LOAD TRAINED GENERATOR FROM STEP2
# -----------------------------------------
generator_path = f"{MODELS_DIR}/generator.pth"
if not os.path.exists(generator_path):
    raise FileNotFoundError(
        f"Missing generator model: {generator_path}. Run step2_train_cgan.py first."
    )

print("\nLoading trained generator...")
generator = Generator().to(device)
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.eval()

# -----------------------------------------
# PRE-GENERATE SYNTHETIC POOLS FOR TABLE ROWS
# -----------------------------------------
synthetic_cache = {}
for generated_n in GENERATED_COUNTS:
    if generated_n == 0:
        continue
    print(f"Generating synthetic pool: {generated_n} per digit")
    gen_imgs, gen_labels = generate_synthetic(generator, generated_n)
    synthetic_cache[generated_n] = (gen_imgs, gen_labels)

# -----------------------------------------
# RUN PROBLEM 6 PART 2 EXPERIMENTS
# -----------------------------------------
results = []

for real_n in REAL_COUNTS:
    real_imgs, real_labels = real_subset_from_pools(real_pools, real_n)

    for generated_n in GENERATED_COUNTS:
        if generated_n == 0:
            train_images = real_imgs
            train_labels = real_labels
        else:
            gen_imgs, gen_labels = synthetic_cache[generated_n]
            train_images = torch.cat([real_imgs, gen_imgs], dim=0)
            train_labels = torch.cat([real_labels, gen_labels], dim=0)

        description = (
            f"Real={real_n}/digit + Generated={generated_n}/digit"
        )
        accuracy, train_time_ms, test_time_ms = train_lenet(
            train_images,
            train_labels,
            test_images,
            test_labels,
            EPOCHS,
            description,
        )

        results.append(
            {
                "real_per_digit": int(real_n),
                "generated_per_digit": int(generated_n),
                "train_size_total": int(train_images.size(0)),
                "test_size_total": int(test_images.size(0)),
                "accuracy": float(accuracy),
                "train_time_ms": float(train_time_ms),
                "test_time_ms": float(test_time_ms),
            }
        )


save_results(results)
plot_heatmap(results)

print("\n" + "=" * 92)
print("PROBLEM 6 PART 2 RESULTS (350/750/1000 REAL)")
print("=" * 92)
print(f"{'Real/digit':>10} {'Gen/digit':>10} {'Train size':>12} {'Accuracy':>10} {'Train ms':>12} {'Test ms':>10}")
print("-" * 92)
for row in results:
    print(
        f"{row['real_per_digit']:>10} "
        f"{row['generated_per_digit']:>10} "
        f"{row['train_size_total']:>12} "
        f"{row['accuracy']:>9.2f}% "
        f"{row['train_time_ms']:>12.1f} "
        f"{row['test_time_ms']:>10.1f}"
    )
print("-" * 92)
print(f"Saved JSON: {PART2_JSON}")
print(f"Saved CSV : {PART2_CSV}")
print(f"Saved plot: {PART2_HEATMAP}")
print("\nStep 4 complete.")