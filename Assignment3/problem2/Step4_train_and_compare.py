"""
Problem 2 - Step 4 (v2): Train LeNet-5 on All Synthetic Sets + Compare
=======================================================================
Point 6: Train LeNet-5 using:
  - 350 real examples per digit
  - Plus the selected GAN-generated samples (from 5-run pooled sets)

Configurations compared:
  1. Baseline-350   : 350 real/digit only
  2. Baseline-1000  : 1000 real/digit only  (upper bound reference)
  3. Real + Set A   : 350 real + all GAN (5 runs × 1000/digit = 5000/digit)
  4. Real + Set B   : 350 real + high-confidence GAN (conf ≥ 0.9)
  5. Real + Set C   : 350 real + mid-confidence GAN  (0.6 ≤ conf < 0.9)

All models evaluated on the same held-out test set.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
# GPU
# ─────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print("=" * 65)
print("  Step 4 v2 — LeNet-5: 350 Real + GAN Samples (5 runs)")
print("=" * 65)
print(f"  GPU : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT    = "/kaggle/input/datasets/mohamedgamal07/reduced-mnist/Reduced MNIST Data"
TRAIN_DIR    = os.path.join(DATA_ROOT, "Reduced Trainging data")
TEST_DIR     = os.path.join(DATA_ROOT, "Reduced Testing data")
NUM_CLASSES  = 10
IMG_SIZE     = 32
LENET_EPOCHS = 40
BATCH_SIZE   = 128
LR           = 1e-3
SEED         = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
base_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ─────────────────────────────────────────────
# REAL DATA LOADER
# ─────────────────────────────────────────────
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
                self.labels.append(torch.tensor(digit, dtype=torch.long))
    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i], self.labels[i]

# ─────────────────────────────────────────────
# LENET-5
# ─────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),  nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5, 120), nn.Tanh(),
            nn.Linear(120, 84),     nn.Tanh(),
            nn.Linear(84, NUM_CLASSES),
        )
    def forward(self, x): return self.classifier(self.features(x))

# ─────────────────────────────────────────────
# TRAINING + EVAL HELPER
# ─────────────────────────────────────────────
def train_lenet(train_loader, test_loader, tag, epochs=LENET_EPOCHS):
    """
    Train one LeNet-5 from scratch and return (final_acc, acc_history, model).
    Each call resets the seed → fair comparison across configs.
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = LeNet5().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    crit  = nn.CrossEntropyLoss()
    hist  = []

    for ep in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE) if isinstance(labels, torch.Tensor) \
                     else torch.tensor(labels).to(DEVICE)
            opt.zero_grad()
            crit(model(imgs), labels).backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.to(DEVICE) if isinstance(labels, torch.Tensor) \
                         else torch.tensor(labels).to(DEVICE)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        hist.append(100 * correct / total)

    final_acc = hist[-1]
    train_size = len(train_loader.dataset)
    print(f"  [{tag:38s}]  Acc: {final_acc:.2f}%  (train: {train_size:,})")
    return final_acc, hist, model

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("\n  Loading datasets ...")

# Point 6: 350 real examples per digit
real350_ds  = FolderDataset(TRAIN_DIR, base_tf, max_per_class=350)
real1000_ds = FolderDataset(TRAIN_DIR, base_tf, max_per_class=1000)  # upper bound ref
test_ds     = FolderDataset(TEST_DIR,  base_tf)

test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                         num_workers=0, pin_memory=True)

# Load GAN sets saved by Step 3 (pooled from 5 runs)
def load_synth(path):
    d = torch.load(path, map_location="cpu")
    return TensorDataset(d["imgs"], d["labels"])

synth_A = load_synth("set_A.pt")
synth_B = load_synth("set_B.pt")
synth_C = load_synth("set_C.pt")

print(f"  Real-350  : {len(real350_ds):,}   ← Point 6 base set")
print(f"  Real-1000 : {len(real1000_ds):,}   ← upper bound reference only")
print(f"  Test      : {len(test_ds):,}")
print(f"  Synth A   : {len(synth_A):,}   (all 5 runs × 1000/digit)")
print(f"  Synth B   : {len(synth_B):,}   (conf ≥ 0.9  filtered)")
print(f"  Synth C   : {len(synth_C):,}   (0.6 ≤ conf < 0.9)")

def make_loader(ds, shuffle=True):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=False)

# ─────────────────────────────────────────────
# TRAIN ALL 5 CONFIGURATIONS
# Point 6: configs 3-5 all use 350 real + GAN selected samples
# ─────────────────────────────────────────────
print(f"\n  Training all configurations ({LENET_EPOCHS} epochs each) ...")
print(f"  {'─'*65}")

results = {}

# 1. Baseline-350 (real only — no GAN)
acc, hist, _ = train_lenet(make_loader(real350_ds), test_loader,
                            "Baseline-350 (real only)")
results["Baseline\n350 real"] = (acc, hist, len(real350_ds))

# 2. Baseline-1000 (real only upper bound — no GAN)
acc, hist, _ = train_lenet(make_loader(real1000_ds), test_loader,
                            "Baseline-1000 (real only)")
results["Baseline\n1000 real"] = (acc, hist, len(real1000_ds))

# 3. Point 6 — 350 real + Set A (ALL 5-run GAN samples)
combined_A = ConcatDataset([real350_ds, synth_A])
acc, hist, _ = train_lenet(make_loader(combined_A), test_loader,
                            "350 real + Set A (all GAN, 5 runs)")
results["350 real\n+ Set A"] = (acc, hist, len(combined_A))

# 4. Point 6 — 350 real + Set B (high-confidence GAN)
combined_B = ConcatDataset([real350_ds, synth_B])
acc, hist, _ = train_lenet(make_loader(combined_B), test_loader,
                            "350 real + Set B (conf≥0.9)")
results["350 real\n+ Set B"] = (acc, hist, len(combined_B))

# 5. Point 6 — 350 real + Set C (mid-confidence GAN)
combined_C = ConcatDataset([real350_ds, synth_C])
acc, hist, _ = train_lenet(make_loader(combined_C), test_loader,
                            "350 real + Set C (0.6-0.9)")
results["350 real\n+ Set C"] = (acc, hist, len(combined_C))

# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────
print(f"\n  {'─'*70}")
print(f"  {'Configuration':<35} | {'Train Size':>10} | {'Test Acc':>10} | {'vs Baseline':>12}")
print(f"  {'─'*70}")
baseline_acc = results["Baseline\n350 real"][0]
for name, (acc, _, size) in results.items():
    label = name.replace("\n", " ")
    delta = acc - baseline_acc
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<35} | {size:>10,} | {acc:>9.2f}% | {sign}{delta:>+10.2f}%")
print(f"  {'─'*70}")

# ─────────────────────────────────────────────
# PLOT 1: Bar chart + Learning curves
# ─────────────────────────────────────────────
labels_list = list(results.keys())
accs        = [v[0] for v in results.values()]
sizes       = [v[2] for v in results.values()]

colors = ["#95a5a6",   # Baseline-350  (gray)
          "#2ecc71",   # Baseline-1000 (green)
          "#3498db",   # Set A (blue)
          "#e74c3c",   # Set B (red)
          "#f39c12"]   # Set C (orange)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

bars = axes[0].bar(labels_list, accs, color=colors, edgecolor="white", width=0.55)
axes[0].set_ylabel("Test Accuracy (%)")
axes[0].set_title("LeNet-5 Test Accuracy\n350 Real + GAN Sets (5 runs)", fontweight="bold")
axes[0].set_ylim(min(accs) - 2, 100)
axes[0].yaxis.set_minor_locator(mticker.AutoMinorLocator())
axes[0].grid(alpha=0.3, axis="y")
axes[0].axhline(y=baseline_acc,
                color="#95a5a6", linestyle="--", lw=1.2,
                label=f"350-real baseline ({baseline_acc:.2f}%)")
axes[0].axhline(y=results["Baseline\n1000 real"][0],
                color="#2ecc71", linestyle="--", lw=1.2,
                label=f"1000-real upper ({results['Baseline\n1000 real'][0]:.2f}%)")
axes[0].legend(fontsize=8)
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{acc:.2f}%", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")

epoch_x = range(1, LENET_EPOCHS + 1)
curve_styles = [
    ("#95a5a6", "--", "Baseline-350"),
    ("#2ecc71", "--", "Baseline-1000"),
    ("#3498db", "-",  "350 + Set A (all)"),
    ("#e74c3c", "-",  "350 + Set B (≥0.9)"),
    ("#f39c12", "-",  "350 + Set C (0.6-0.9)"),
]
for (name, (acc, hist, _)), (color, ls, label) in zip(results.items(), curve_styles):
    axes[1].plot(epoch_x, hist, color=color, linestyle=ls, lw=1.8, label=label)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy (%)")
axes[1].set_title("Learning Curves\n350 Real + GAN Sets (5 runs)", fontweight="bold")
axes[1].legend(fontsize=8.5)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(70, 100)

plt.tight_layout()
plt.savefig("step4_accuracy_comparison.png", dpi=130, bbox_inches="tight")
plt.show()
print("\n  [✓] Saved → step4_accuracy_comparison.png")

# ─────────────────────────────────────────────
# PLOT 2: Per-class accuracy — baseline vs best GAN
# ─────────────────────────────────────────────
def per_class_acc(model, loader):
    model.eval()
    correct = [0] * NUM_CLASSES
    total   = [0] * NUM_CLASSES
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE) if isinstance(labels, torch.Tensor) \
                     else torch.tensor(labels).to(DEVICE)
            preds = model(imgs).argmax(1)
            for d in range(NUM_CLASSES):
                mask = (labels == d)
                correct[d] += (preds[mask] == d).sum().item()
                total[d]   += mask.sum().item()
    return [100 * correct[d] / total[d] if total[d] > 0 else 0
            for d in range(NUM_CLASSES)]

print("\n  Computing per-class accuracy (baseline vs best GAN) ...")

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
_, _, model_base = train_lenet(make_loader(real350_ds), test_loader,
                                "Baseline-350 [per-class eval]")

gan_configs  = {k: v for k, v in results.items() if "Set" in k}
best_name    = max(gan_configs, key=lambda k: gan_configs[k][0])
best_ds_map  = {
    "350 real\n+ Set A": combined_A,
    "350 real\n+ Set B": combined_B,
    "350 real\n+ Set C": combined_C,
}
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
_, _, model_best = train_lenet(
    make_loader(best_ds_map[best_name]), test_loader,
    f"Best GAN [{best_name.replace(chr(10),' ')}] [per-class eval]")

pc_base = per_class_acc(model_base, test_loader)
pc_best = per_class_acc(model_best, test_loader)

x = np.arange(NUM_CLASSES)
w = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, pc_base, w, label="Baseline-350 (real only)",
       color="#95a5a6", edgecolor="white")
ax.bar(x + w/2, pc_best, w,
       label=f"Best: {best_name.replace(chr(10),' ')} (350 real + GAN 5 runs)",
       color="#e74c3c", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(range(NUM_CLASSES))
ax.set_xlabel("Digit Class")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-Class Accuracy: Baseline-350 vs Best GAN Config (5 runs)",
             fontweight="bold")
ax.legend()
ax.set_ylim(70, 101)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("step4_perclass_accuracy.png", dpi=130, bbox_inches="tight")
plt.show()
print("  [✓] Saved → step4_perclass_accuracy.png")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY — PROBLEM 2  (Point 6)")
print("=" * 70)
print(f"  {'Configuration':<35} | {'Train Size':>10} | {'Test Acc':>10} | {'Δ vs 350':>10}")
print(f"  {'─'*68}")
for name, (acc, _, size) in results.items():
    label = name.replace("\n", " ")
    delta = acc - baseline_acc
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<35} | {size:>10,} | {acc:>9.2f}% | {sign}{delta:.2f}%")
print(f"  {'─'*68}")

best_gan_acc = gan_configs[best_name][0]
real1000_acc = results["Baseline\n1000 real"][0]
gap_closed   = (best_gan_acc - baseline_acc) / max(real1000_acc - baseline_acc, 1e-6) * 100

print(f"\n  Best GAN strategy  : {best_name.replace(chr(10), ' ')}")
print(f"  Best GAN accuracy  : {best_gan_acc:.2f}%")
print(f"  Gap to 1000-real   : {real1000_acc - best_gan_acc:.2f}%")
print(f"  Gap closed by GAN  : {gap_closed:.1f}% of the 350→1000 gap")
print("=" * 70)
print("\n✅  Step 4 complete — ready for final report (Step 5).")