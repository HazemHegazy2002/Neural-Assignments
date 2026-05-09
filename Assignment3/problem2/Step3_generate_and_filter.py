"""
Problem 2 - Step 3 (v2): Generate Synthetic Samples + Confidence Filtering
===========================================================================
Point 4: Generate 5× (1000 samples per digit per run) = 5000 per digit total.
         Each run starts with DIFFERENT random seeds → better sample variety.

Point 6: Filter with LeNet-5 confidence to build Sets A, B, C for Step 4.

Sets produced:
  Set A : all 50,000 generated samples  (5000 per digit)
  Set B : high-confidence  conf >= 0.9
  Set C : mid-confidence   0.6 <= conf < 0.9
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# GPU CHECK
# ─────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print("=" * 65)
print("  Step 3 v2 — Generate (5 runs) & Filter Synthetic Samples")
print("=" * 65)
print(f"  GPU : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT   = "/kaggle/input/datasets/mohamedgamal07/reduced-mnist/Reduced MNIST Data"
TRAIN_DIR   = os.path.join(DATA_ROOT, "Reduced Trainging data")
TEST_DIR    = os.path.join(DATA_ROOT, "Reduced Testing data")
NUM_CLASSES = 10
IMG_SIZE    = 32
LATENT_DIM  = 100
EMBED_DIM   = 50

NUM_RUNS          = 5      # ← Point 4: 5 independent generation runs
SAMPLES_PER_DIGIT = 1000   # per run  →  5000 per digit total across all runs
BATCH_GEN         = 500
BASE_SEED         = 42

# Each run gets its own seed so random z vectors differ
RUN_SEEDS = [BASE_SEED + i * 1000 for i in range(NUM_RUNS)]
# e.g. [42, 1042, 2042, 3042, 4042]

print(f"\n  Runs         : {NUM_RUNS}")
print(f"  Samples/digit/run : {SAMPLES_PER_DIGIT}")
print(f"  Total/digit  : {NUM_RUNS * SAMPLES_PER_DIGIT:,}")
print(f"  Grand total  : {NUM_RUNS * SAMPLES_PER_DIGIT * NUM_CLASSES:,}")
print(f"  Run seeds    : {RUN_SEEDS}")

# ─────────────────────────────────────────────
# RE-DEFINE GENERATOR (must match Step 2)
# ─────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.project = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 512 * 4 * 4), nn.ReLU(True))
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1, bias=False), nn.Tanh())
    def forward(self, z, labels):
        e = self.label_embed(labels)
        return self.conv(self.project(torch.cat([z, e], 1)).view(-1, 512, 4, 4))

G = Generator().to(DEVICE)
G.load_state_dict(torch.load("generator.pth", map_location=DEVICE))
G.eval()
print("\n  [✓] Generator loaded from generator.pth")

# ─────────────────────────────────────────────
# LENET-5 CLASSIFIER  (confidence filter)
# ─────────────────────────────────────────────
class LeNet5(nn.Module):
    """Standard LeNet-5 for 32×32 grayscale input."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),   nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),  nn.Tanh(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5, 120), nn.Tanh(),
            nn.Linear(120, 84),     nn.Tanh(),
            nn.Linear(84, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ─────────────────────────────────────────────
# LOAD REAL DATA — train LeNet-5 as confidence scorer
# ─────────────────────────────────────────────
base_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

class FolderDataset(Dataset):
    def __init__(self, root, transform, max_per_class=None):
        self.samples, self.labels = [], []
        for digit in range(NUM_CLASSES):
            folder = os.path.join(root, str(digit))
            files  = sorted([f for f in os.listdir(folder)
                             if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))])
            if max_per_class:
                files = files[:max_per_class]
            for fname in files:
                img = Image.open(os.path.join(folder, fname)).convert("L")
                self.samples.append(transform(img))
                self.labels.append(digit)
    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i], self.labels[i]

print("\n  Loading real train / test data for LeNet-5 ...")
train_ds = FolderDataset(TRAIN_DIR, base_tf, max_per_class=350)
test_ds  = FolderDataset(TEST_DIR,  base_tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=0, pin_memory=True)
print(f"  Train: {len(train_ds)}  |  Test: {len(test_ds)}")

# Train LeNet-5 on 350 real/digit
print("\n  Training LeNet-5 on 350 real images/digit ...")
torch.manual_seed(BASE_SEED)
torch.cuda.manual_seed_all(BASE_SEED)
lenet = LeNet5().to(DEVICE)
opt   = torch.optim.Adam(lenet.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
crit  = nn.CrossEntropyLoss()

LENET_EPOCHS = 40
for ep in range(1, LENET_EPOCHS + 1):
    lenet.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), torch.tensor(labels).to(DEVICE)
        opt.zero_grad()
        crit(lenet(imgs), labels).backward()
        opt.step()
    sched.step()
    if ep % 10 == 0 or ep == 1:
        lenet.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), torch.tensor(labels).to(DEVICE)
                correct += (lenet(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        print(f"    Epoch {ep:>3} | Test Acc: {100*correct/total:.2f}%")

lenet.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), torch.tensor(labels).to(DEVICE)
        correct += (lenet(imgs).argmax(1) == labels).sum().item()
        total   += labels.size(0)
lenet_baseline_acc = 100 * correct / total
print(f"\n  LeNet-5 baseline accuracy (350 real/digit): {lenet_baseline_acc:.2f}%")
torch.save(lenet.state_dict(), "lenet5_baseline.pth")
print("  [✓] Saved → lenet5_baseline.pth")

# ─────────────────────────────────────────────
# GENERATE — 5 RUNS, EACH WITH DIFFERENT SEED
# ─────────────────────────────────────────────
# Point 4: Each run starts with different random values (different seed)
# so the z vectors sampled from N(0,1) are different each time → more diverse
# generated samples.

print(f"\n  ── Generating {NUM_RUNS} runs × {SAMPLES_PER_DIGIT} samples/digit ──")

all_imgs_list   = []
all_labels_list = []
all_run_ids     = []   # track which run each sample came from

G.eval()
for run_idx, seed in enumerate(RUN_SEEDS):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    print(f"\n  Run {run_idx + 1}/{NUM_RUNS}  (seed={seed})")

    run_imgs   = []
    run_labels = []

    with torch.no_grad():
        for digit in range(NUM_CLASSES):
            digit_imgs = []
            remaining  = SAMPLES_PER_DIGIT
            while remaining > 0:
                bsz  = min(BATCH_GEN, remaining)
                z    = torch.randn(bsz, LATENT_DIM, device=DEVICE)
                lbl  = torch.full((bsz,), digit, dtype=torch.long, device=DEVICE)
                imgs = G(z, lbl).cpu()
                digit_imgs.append(imgs)
                remaining -= bsz
            digit_imgs = torch.cat(digit_imgs, dim=0)[:SAMPLES_PER_DIGIT]
            run_imgs.append(digit_imgs)
            run_labels.extend([digit] * SAMPLES_PER_DIGIT)
            print(f"    Digit {digit}: {SAMPLES_PER_DIGIT} samples  ✓", end="\r")

    run_imgs_t   = torch.cat(run_imgs, dim=0)
    run_labels_t = torch.tensor(run_labels)

    all_imgs_list.append(run_imgs_t)
    all_labels_list.append(run_labels_t)
    all_run_ids.append(torch.full((len(run_imgs_t),), run_idx, dtype=torch.long))

    print(f"    Run {run_idx+1} complete: {len(run_imgs_t):,} images generated")

# Pool all runs together
all_imgs   = torch.cat(all_imgs_list,   dim=0)   # (50000, 1, 32, 32)
all_labels = torch.cat(all_labels_list, dim=0)   # (50000,)
all_run_ids = torch.cat(all_run_ids,    dim=0)   # (50000,)

print(f"\n  Total generated across all runs : {len(all_imgs):,} images")
print(f"  Per digit                        : {len(all_imgs) // NUM_CLASSES:,}")

# ─────────────────────────────────────────────
# COMPUTE CONFIDENCE SCORES
# ─────────────────────────────────────────────
print("\n  Computing LeNet-5 confidence scores ...")
lenet.eval()
all_conf  = []
all_preds = []

gen_dataset = TensorDataset(all_imgs, all_labels)
gen_loader  = DataLoader(gen_dataset, batch_size=512, shuffle=False,
                         num_workers=0, pin_memory=True)

with torch.no_grad():
    for imgs, _ in gen_loader:
        logits = lenet(imgs.to(DEVICE))
        probs  = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        all_conf.append(conf.cpu())
        all_preds.append(pred.cpu())

all_conf  = torch.cat(all_conf)
all_preds = torch.cat(all_preds)

print(f"  Confidence stats (pooled {NUM_RUNS} runs):")
print(f"    Mean   : {all_conf.mean():.4f}")
print(f"    Median : {all_conf.median():.4f}")
print(f"    Min    : {all_conf.min():.4f}")
print(f"    Max    : {all_conf.max():.4f}")

# ─────────────────────────────────────────────
# CREATE SETS A, B, C
# ─────────────────────────────────────────────
mask_B = all_conf >= 0.9
mask_C = (all_conf >= 0.6) & (all_conf < 0.9)

set_A_imgs,  set_A_labels  = all_imgs,          all_labels
set_B_imgs,  set_B_labels  = all_imgs[mask_B],  all_labels[mask_B]
set_C_imgs,  set_C_labels  = all_imgs[mask_C],  all_labels[mask_C]

print(f"\n  ┌──────────────────────────────────────────────────┐")
print(f"  │  SYNTHETIC DATASET SUMMARY  ({NUM_RUNS} runs pooled)      │")
print(f"  ├──────────────────────────────────────────────────┤")
print(f"  │  Set A (all generated)   : {len(set_A_imgs):>8,} images     │")
print(f"  │  Set B (conf >= 0.9)     : {len(set_B_imgs):>8,} images     │")
print(f"  │  Set C (0.6 <= conf<0.9) : {len(set_C_imgs):>8,} images     │")
print(f"  └──────────────────────────────────────────────────┘")

print(f"\n  Per-digit counts (across all {NUM_RUNS} runs):")
print(f"  {'Digit':>6} | {'Set A':>8} | {'Set B (≥0.9)':>12} | {'Set C (0.6-0.9)':>16}")
print(f"  " + "-" * 52)
for d in range(NUM_CLASSES):
    na = (set_A_labels == d).sum().item()
    nb = (set_B_labels == d).sum().item()
    nc = (set_C_labels == d).sum().item()
    print(f"  {d:>6} | {na:>8} | {nb:>12} | {nc:>16}")

# ─────────────────────────────────────────────
# SAVE DATASETS
# ─────────────────────────────────────────────
torch.save({"imgs": set_A_imgs, "labels": set_A_labels, "conf": all_conf,
            "run_ids": all_run_ids},
           "set_A.pt")
torch.save({"imgs": set_B_imgs, "labels": set_B_labels, "conf": all_conf[mask_B],
            "run_ids": all_run_ids[mask_B]},
           "set_B.pt")
torch.save({"imgs": set_C_imgs, "labels": set_C_labels, "conf": all_conf[mask_C],
            "run_ids": all_run_ids[mask_C]},
           "set_C.pt")
print("\n  [✓] Saved → set_A.pt, set_B.pt, set_C.pt")

# ─────────────────────────────────────────────
# VISUALISATION 1: Confidence Histogram
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(all_conf.numpy(), bins=60, color="steelblue", edgecolor="white", lw=0.4)
axes[0].axvline(0.9, color="green",  lw=2, linestyle="--", label="Set B threshold (0.9)")
axes[0].axvline(0.6, color="orange", lw=2, linestyle="--", label="Set C threshold (0.6)")
axes[0].set_xlabel("Confidence (max softmax)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Confidence Distribution — {len(all_imgs):,} Samples ({NUM_RUNS} runs)")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

mean_conf_per_digit = [all_conf[all_labels == d].mean().item() for d in range(NUM_CLASSES)]
colors = ["#2ecc71" if c >= 0.9 else "#e67e22" if c >= 0.6 else "#e74c3c"
          for c in mean_conf_per_digit]
axes[1].bar(range(NUM_CLASSES), mean_conf_per_digit, color=colors, edgecolor="white")
axes[1].axhline(0.9, color="green",  lw=1.5, linestyle="--", label="0.9 threshold")
axes[1].axhline(0.6, color="orange", lw=1.5, linestyle="--", label="0.6 threshold")
axes[1].set_xticks(range(NUM_CLASSES))
axes[1].set_xlabel("Digit")
axes[1].set_ylabel("Mean Confidence")
axes[1].set_title("Mean Confidence per Digit (pooled runs)")
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 1)
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("step3_confidence_histogram.png", dpi=120)
plt.show()
print("  [✓] Saved → step3_confidence_histogram.png")

# ─────────────────────────────────────────────
# VISUALISATION 2: Per-run confidence comparison
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for run_idx in range(NUM_RUNS):
    run_mask = (all_run_ids == run_idx)
    run_conf = all_conf[run_mask]
    ax.hist(run_conf.numpy(), bins=40, alpha=0.55,
            label=f"Run {run_idx+1} (seed={RUN_SEEDS[run_idx]})",
            edgecolor="none")
ax.axvline(0.9, color="green",  lw=2, linestyle="--", label="0.9 threshold")
ax.axvline(0.6, color="orange", lw=2, linestyle="--", label="0.6 threshold")
ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
ax.set_title("Confidence Distribution per Run — Different Seeds Produce Varied Samples")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("step3_per_run_confidence.png", dpi=120)
plt.show()
print("  [✓] Saved → step3_per_run_confidence.png")

# ─────────────────────────────────────────────
# VISUALISATION 3: Sample grid for each set
# ─────────────────────────────────────────────
def show_set_grid(imgs, labels, confs, title, fname, n_per_digit=8):
    fig, axes = plt.subplots(NUM_CLASSES, n_per_digit,
                             figsize=(n_per_digit * 1.3, NUM_CLASSES * 1.4))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for digit in range(NUM_CLASSES):
        idx = (labels == digit).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            for j in range(n_per_digit):
                axes[digit, j].axis("off")
            axes[digit, 0].set_ylabel(str(digit), fontsize=9,
                                      rotation=0, labelpad=18, va="center")
            continue
        pick = idx[:n_per_digit]
        for j, ii in enumerate(pick):
            img = (imgs[ii] * 0.5 + 0.5).clamp(0, 1).squeeze()
            axes[digit, j].imshow(img, cmap="gray")
            axes[digit, j].set_title(f"{confs[ii]:.2f}", fontsize=6, pad=1)
            axes[digit, j].axis("off")
        for j in range(len(pick), n_per_digit):
            axes[digit, j].axis("off")
        axes[digit, 0].set_ylabel(str(digit), fontsize=9,
                                  rotation=0, labelpad=18, va="center")
    plt.tight_layout()
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  [✓] Saved → {fname}")

show_set_grid(set_A_imgs, set_A_labels, all_conf,
              f"Set A — All Generated ({NUM_RUNS} runs, top 8 per digit)",
              "step3_setA_grid.png")

if len(set_B_imgs) > 0:
    show_set_grid(set_B_imgs, set_B_labels, all_conf[mask_B],
                  "Set B — High Confidence ≥ 0.9 (top 8 per digit)",
                  "step3_setB_grid.png")
else:
    print("  ⚠️  Set B is empty — no samples with confidence ≥ 0.9")

if len(set_C_imgs) > 0:
    show_set_grid(set_C_imgs, set_C_labels, all_conf[mask_C],
                  "Set C — Mid Confidence 0.6 ≤ conf < 0.9 (top 8 per digit)",
                  "step3_setC_grid.png")
else:
    print("  ⚠️  Set C is empty — no samples with 0.6 ≤ confidence < 0.9")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 3 SUMMARY")
print("=" * 65)
print(f"  Generation runs        : {NUM_RUNS}  (seeds: {RUN_SEEDS})")
print(f"  Samples per digit/run  : {SAMPLES_PER_DIGIT:,}")
print(f"  Total per digit        : {NUM_RUNS * SAMPLES_PER_DIGIT:,}")
print(f"  LeNet-5 baseline acc   : {lenet_baseline_acc:.2f}%  (350 real/digit)")
print(f"  Total generated        : {len(set_A_imgs):,}")
print(f"  Set A (all)            : {len(set_A_imgs):,}")
print(f"  Set B (conf ≥ 0.9)     : {len(set_B_imgs):,}")
print(f"  Set C (0.6–0.9)        : {len(set_C_imgs):,}")
print(f"  Mean confidence        : {all_conf.mean():.4f}")
print("=" * 65)
print("\n✅  Step 3 complete — ready for LeNet-5 training on augmented sets (Step 4).")