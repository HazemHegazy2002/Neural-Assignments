"""
Problem 1 - Step 3 (cVAE): Generate Synthetic Samples + Confidence Filtering
=============================================================================
For each digit, run the cVAE Decoder 5 times × 1000 samples = 5,000 per digit.
Each run uses a different random seed so z ~ N(0,I) is sampled from a different
region of the latent space, producing more diverse generated images.
Total generated: 5 runs × 10 digits × 1000 = 50,000 images.

Pass them through a pre-trained LeNet-5 classifier to get softmax confidence.
Create three filtered datasets:
  Set A : all 50,000 generated samples  (5,000 per digit)
  Set B : high-confidence  confidence >= 0.9
  Set C : mid-confidence   0.6 <= confidence < 0.9
"""



import os
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

print("=" * 62)
print("  Step 3 (cVAE) — Generate & Filter Synthetic Samples")
print("=" * 62)
print(f"  GPU : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT         = "/kaggle/input/datasets/mohamedgamal07/reduced-mnist/Reduced MNIST Data"
TRAIN_DIR         = os.path.join(DATA_ROOT, "Reduced Trainging data")
TEST_DIR          = os.path.join(DATA_ROOT, "Reduced Testing data")
NUM_CLASSES       = 10
IMG_SIZE          = 32
LATENT_DIM        = 128   # ← must match cVAE Step 2
EMBED_DIM         = 50    # ← must match cVAE Step 2
NUM_RUNS          = 5     # number of independent generation runs per digit
SAMPLES_PER_RUN   = 1000  # samples generated per digit per run
SAMPLES_PER_DIGIT = NUM_RUNS * SAMPLES_PER_RUN   # 5,000 per digit → 50,000 total
BATCH_GEN         = 500   # generation batch size
SEED              = 42    # base seed; each run uses SEED + run_idx for diversity

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# RE-DEFINE DECODER  (must match Step 2 exactly)
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    """
    cVAE Decoder: (z, label) → reconstructed image [B, 1, 32, 32]
    Identical to the Decoder defined in Step 2.
    """
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.project = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 512 * 4 * 4),
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        e = self.label_embed(labels)                        # [B, EMBED_DIM]
        x = self.project(torch.cat([z, e], dim=1))         # [B, 512*4*4]
        return self.conv(x.view(-1, 512, 4, 4))            # [B, 1, 32, 32]


# Load trained decoder weights
decoder = Decoder().to(DEVICE)
decoder.load_state_dict(torch.load("decoder.pth", map_location=DEVICE))
decoder.eval()
print("\n  [✓] cVAE Decoder loaded from decoder.pth")

# ─────────────────────────────────────────────
# LENET-5 CLASSIFIER  
# ─────────────────────────────────────────────
class LeNet5(nn.Module):
    """Standard LeNet-5 adapted for 32×32 grayscale input."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6,  5), nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84),          nn.Tanh(),
            nn.Linear(84, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ─────────────────────────────────────────────
# LOAD REAL DATA FOR LENET TRAINING  
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
    def __len__(self):         return len(self.samples)
    def __getitem__(self, i):  return self.samples[i], self.labels[i]

print("\n  Loading real train / test data for LeNet-5 ...")
train_ds     = FolderDataset(TRAIN_DIR, base_tf, max_per_class=350)
test_ds      = FolderDataset(TEST_DIR,  base_tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
print(f"  Train: {len(train_ds)}  |  Test: {len(test_ds)}")

# ─────────────────────────────────────────────
# TRAIN LENET-5  
# ─────────────────────────────────────────────
print("\n  Training LeNet-5 on 350 real images/digit ...")
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
        loss = crit(lenet(imgs), labels)
        loss.backward()
        opt.step()
    sched.step()
    if ep % 10 == 0 or ep == 1:
        lenet.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), torch.tensor(labels).to(DEVICE)
                preds = lenet(imgs).argmax(1)
                correct += (preds == labels).sum().item()
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


print(f"\n  Generating {NUM_RUNS} runs × {SAMPLES_PER_RUN} samples/digit "
      f"= {SAMPLES_PER_DIGIT} per digit ({SAMPLES_PER_DIGIT * NUM_CLASSES:,} total) ...")
all_imgs   = []
all_labels = []

decoder.eval()
with torch.no_grad():
    for run_idx in range(NUM_RUNS):
        # Each run uses a different seed → different z vectors → more diverse samples
        run_seed = SEED + run_idx
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        np.random.seed(run_seed)

        print(f"\n  ── Run {run_idx + 1}/{NUM_RUNS}  (seed={run_seed}) ──")
        for digit in range(NUM_CLASSES):
            imgs_digit = []
            remaining  = SAMPLES_PER_RUN
            while remaining > 0:
                bsz  = min(BATCH_GEN, remaining)
                # Sample z ~ N(0, I) — prior of the VAE
                z    = torch.randn(bsz, LATENT_DIM, device=DEVICE)
                lbl  = torch.full((bsz,), digit, dtype=torch.long, device=DEVICE)
                imgs = decoder(z, lbl).cpu()
                imgs_digit.append(imgs)
                remaining -= bsz
            imgs_digit = torch.cat(imgs_digit, dim=0)[:SAMPLES_PER_RUN]
            all_imgs.append(imgs_digit)
            all_labels.extend([digit] * SAMPLES_PER_RUN)
            print(f"    Digit {digit}: {SAMPLES_PER_RUN} images  ✓")

all_imgs   = torch.cat(all_imgs, dim=0)   # (50000, 1, 32, 32)
all_labels = torch.tensor(all_labels)     # (50000,)
print(f"\n  Total generated: {len(all_imgs):,} images  "
      f"({NUM_RUNS} runs × {SAMPLES_PER_RUN}/digit × {NUM_CLASSES} digits)")

# ─────────────────────────────────────────────
# COMPUTE CONFIDENCE SCORES  
# ─────────────────────────────────────────────
print("\n  Computing confidence scores via LeNet-5 ...")
lenet.eval()
all_conf  = []
all_preds = []

gen_dataset = TensorDataset(all_imgs, all_labels)
gen_loader  = DataLoader(gen_dataset, batch_size=512, shuffle=False,
                         num_workers=0, pin_memory=True)

with torch.no_grad():
    for imgs, _ in gen_loader:
        imgs   = imgs.to(DEVICE)
        logits = lenet(imgs)
        probs  = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        all_conf.append(conf.cpu())
        all_preds.append(pred.cpu())

all_conf  = torch.cat(all_conf)    # (50000,)
all_preds = torch.cat(all_preds)   # (50000,)

print(f"  Confidence stats:")
print(f"    Mean  : {all_conf.mean():.4f}")
print(f"    Median: {all_conf.median():.4f}")
print(f"    Min   : {all_conf.min():.4f}")
print(f"    Max   : {all_conf.max():.4f}")

# ─────────────────────────────────────────────
# CREATE SETS A, B, C  
# ─────────────────────────────────────────────
mask_B = all_conf >= 0.9
mask_C = (all_conf >= 0.6) & (all_conf < 0.9)

set_A_imgs,  set_A_labels  = all_imgs,         all_labels
set_B_imgs,  set_B_labels  = all_imgs[mask_B], all_labels[mask_B]
set_C_imgs,  set_C_labels  = all_imgs[mask_C], all_labels[mask_C]

print(f"\n  ┌─────────────────────────────────────────────┐")
print(f"  │  SYNTHETIC DATASET SUMMARY                   │")
print(f"  ├─────────────────────────────────────────────┤")
print(f"  │  Runs × Samples/digit : {NUM_RUNS} × {SAMPLES_PER_RUN} = {SAMPLES_PER_DIGIT:,}/digit    │")
print(f"  │  Set A (all generated)   : {len(set_A_imgs):>6} images   │")
print(f"  │  Set B (conf >= 0.9)     : {len(set_B_imgs):>6} images   │")
print(f"  │  Set C (0.6 <= conf<0.9) : {len(set_C_imgs):>6} images   │")
print(f"  └─────────────────────────────────────────────┘")

print(f"\n  Per-digit counts:")
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
torch.save({"imgs": set_A_imgs, "labels": set_A_labels, "conf": all_conf},
           "set_A.pt")
torch.save({"imgs": set_B_imgs, "labels": set_B_labels, "conf": all_conf[mask_B]},
           "set_B.pt")
torch.save({"imgs": set_C_imgs, "labels": set_C_labels, "conf": all_conf[mask_C]},
           "set_C.pt")
print("\n  [✓] Saved → set_A.pt, set_B.pt, set_C.pt")

# ─────────────────────────────────────────────
# VISUALISATION 1: Confidence Histogram  
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(all_conf.numpy(), bins=50, color="steelblue", edgecolor="white", lw=0.4)
axes[0].axvline(0.9, color="green",  lw=2, linestyle="--", label="Set B threshold (0.9)")
axes[0].axvline(0.6, color="orange", lw=2, linestyle="--", label="Set C threshold (0.6)")
axes[0].set_xlabel("Confidence (max softmax)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Confidence Distribution — All {len(all_imgs):,} cVAE Generated Samples "
                  f"({NUM_RUNS} runs × {SAMPLES_PER_RUN}/digit)")
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
axes[1].set_title("Mean Confidence per Digit (cVAE)")
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 1)
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("step3_cvae_confidence_histogram.png", dpi=120)
plt.show()
print("  [✓] Saved → step3_cvae_confidence_histogram.png")

# ─────────────────────────────────────────────
# VISUALISATION 2: Sample grids  
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
                axes[digit, j].set_facecolor("black")
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
              "Set A — All cVAE Generated (top 8 per digit, with confidence)",
              "step3_cvae_setA_grid.png")

if len(set_B_imgs) > 0:
    show_set_grid(set_B_imgs, set_B_labels, all_conf[mask_B],
                  "Set B — High Confidence ≥ 0.9 (top 8 per digit)",
                  "step3_cvae_setB_grid.png")
else:
    print("  ⚠️  Set B is empty — no samples with confidence ≥ 0.9")

if len(set_C_imgs) > 0:
    show_set_grid(set_C_imgs, set_C_labels, all_conf[mask_C],
                  "Set C — Mid Confidence 0.6 ≤ conf < 0.9 (top 8 per digit)",
                  "step3_cvae_setC_grid.png")
else:
    print("  ⚠️  Set C is empty — no samples with 0.6 ≤ confidence < 0.9")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 3 SUMMARY  (cVAE)")
print("=" * 62)
print(f"  LeNet-5 baseline acc   : {lenet_baseline_acc:.2f}%  (350 real/digit)")
print(f"  Generation runs        : {NUM_RUNS} × {SAMPLES_PER_RUN}/digit (different seed each run)")
print(f"  Total generated        : {len(set_A_imgs):,}")
print(f"  Set A (all)            : {len(set_A_imgs):,}  ({SAMPLES_PER_DIGIT}/digit)")
print(f"  Set B (conf ≥ 0.9)     : {len(set_B_imgs):,}")
print(f"  Set C (0.6–0.9)        : {len(set_C_imgs):,}")
print(f"  Mean confidence        : {all_conf.mean():.4f}")
print("=" * 62)
print("\n✅  Step 3 complete — ready for LeNet-5 training on augmented sets (Step 4).")