"""
Problem 1 - Step 4 (cVAE): Train LeNet-5 on All Synthetic Sets + Compare
=========================================================================
Trains LeNet-5 on 5 configurations:
  1. Baseline-350   : 350 real/digit only
  2. Baseline-1000  : 1000 real/digit only  (upper bound reference)
  3. Real + Set A   : 350 real + all 10,000 cVAE generated
  4. Real + Set B   : 350 real + high-confidence cVAE samples (≥0.9)
  5. Real + Set C   : 350 real + mid-confidence cVAE samples (0.6–0.9)

All models evaluated on the same 2,000-image test set.
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
print("  Step 4 (cVAE) — LeNet-5 Training on All Configurations")
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
# REAL DATA LOADER HELPER
# ─────────────────────────────────────────────
class FolderDataset(Dataset):
    def __init__(self, root, transform, max_per_class=None):
        self.samples, self.labels = [], []
        for digit in range(NUM_CLASSES):
            folder = os.path.join(root, str(digit))
            files  = sorted([f for f in os.listdir(folder)
                             if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))])
            if max_per_class: files = files[:max_per_class]
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

        # eval every epoch
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.to(DEVICE) if isinstance(labels, torch.Tensor) \
                         else torch.tensor(labels).to(DEVICE)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        acc = 100 * correct / total
        hist.append(acc)

    final_acc = hist[-1]
    print(f"  [{tag:30s}]  Final Test Acc: {final_acc:.2f}%  "
          f"(train size: {len(train_loader.dataset):,})")
    return final_acc, hist, model

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("\n  Loading datasets ...")

# Real sets
real350_ds  = FolderDataset(TRAIN_DIR, base_tf, max_per_class=350)
real1000_ds = FolderDataset(TRAIN_DIR, base_tf, max_per_class=1000)
test_ds     = FolderDataset(TEST_DIR,  base_tf)

test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                         num_workers=0, pin_memory=True)

# Synthetic sets (saved as .pt files in Step 3)
def load_synth(path):
    d = torch.load(path, map_location="cpu")
    return TensorDataset(d["imgs"], d["labels"])

synth_A = load_synth("set_A.pt")
synth_B = load_synth("set_B.pt")
synth_C = load_synth("set_C.pt")

print(f"  Real-350  : {len(real350_ds):,}")
print(f"  Real-1000 : {len(real1000_ds):,}")
print(f"  Test      : {len(test_ds):,}")
print(f"  Synth A   : {len(synth_A):,}")
print(f"  Synth B   : {len(synth_B):,}")
print(f"  Synth C   : {len(synth_C):,}")

def make_loader(ds, shuffle=True):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=False)

# ─────────────────────────────────────────────
# TRAIN ALL 5 CONFIGURATIONS
# ─────────────────────────────────────────────
print(f"\n  Training all configurations ({LENET_EPOCHS} epochs each) ...")
print(f"  {'─'*60}")

results = {}

# 1. Baseline-350
acc, hist, _ = train_lenet(make_loader(real350_ds), test_loader,
                            "Baseline-350 (real only)")
results["Baseline\n350 real"] = (acc, hist, len(real350_ds))

# 2. Baseline-1000
acc, hist, _ = train_lenet(make_loader(real1000_ds), test_loader,
                            "Baseline-1000 (real only)")
results["Baseline\n1000 real"] = (acc, hist, len(real1000_ds))

# 3. Real-350 + Set A
combined_A = ConcatDataset([real350_ds, synth_A])
acc, hist, _ = train_lenet(make_loader(combined_A), test_loader,
                            "Real-350 + Set A (all cVAE)")
results["350 real\n+ Set A"] = (acc, hist, len(combined_A))

# 4. Real-350 + Set B
combined_B = ConcatDataset([real350_ds, synth_B])
acc, hist, _ = train_lenet(make_loader(combined_B), test_loader,
                            "Real-350 + Set B (conf≥0.9)")
results["350 real\n+ Set B"] = (acc, hist, len(combined_B))

# 5. Real-350 + Set C
combined_C = ConcatDataset([real350_ds, synth_C])
acc, hist, _ = train_lenet(make_loader(combined_C), test_loader,
                            "Real-350 + Set C (0.6-0.9)")
results["350 real\n+ Set C"] = (acc, hist, len(combined_C))

# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────
print(f"\n  {'─'*65}")
print(f"  {'Configuration':<30} | {'Train Size':>10} | {'Test Acc':>10} | {'vs Baseline':>12}")
print(f"  {'─'*65}")
baseline_acc = results["Baseline\n350 real"][0]
for name, (acc, _, size) in results.items():
    label = name.replace("\n", " ")
    delta = acc - baseline_acc
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<30} | {size:>10,} | {acc:>9.2f}% | {sign}{delta:>+10.2f}%")
print(f"  {'─'*65}")

# ─────────────────────────────────────────────
# PLOT 1: Bar chart comparison
# ─────────────────────────────────────────────
labels = list(results.keys())
accs   = [v[0] for v in results.values()]
sizes  = [v[2] for v in results.values()]

colors = ["#95a5a6",        # Baseline-350  (gray)
          "#2ecc71",        # Baseline-1000 (green — upper bound)
          "#3498db",        # Set A (blue)
          "#e74c3c",        # Set B (red)
          "#f39c12"]        # Set C (orange)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy bars
bars = axes[0].bar(labels, accs, color=colors, edgecolor="white", width=0.55)
axes[0].set_ylabel("Test Accuracy (%)")
axes[0].set_title("LeNet-5 Test Accuracy — All Configurations (cVAE)", fontweight="bold")
axes[0].set_ylim(min(accs) - 2, 100)
axes[0].yaxis.set_minor_locator(mticker.AutoMinorLocator())
axes[0].grid(alpha=0.3, axis="y")
axes[0].axhline(y=baseline_acc, color="#95a5a6", linestyle="--", lw=1.2,
                label=f"350-real baseline ({baseline_acc:.2f}%)")
axes[0].axhline(y=results["Baseline\n1000 real"][0], color="#2ecc71",
                linestyle="--", lw=1.2,
                label=f"1000-real upper bound ({results['Baseline\n1000 real'][0]:.2f}%)")
axes[0].legend(fontsize=8)
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Learning curves
epoch_x = range(1, LENET_EPOCHS + 1)
curve_styles = [
    ("#95a5a6", "--", "Baseline-350"),
    ("#2ecc71", "--", "Baseline-1000"),
    ("#3498db", "-",  "350 + Set A (cVAE)"),
    ("#e74c3c", "-",  "350 + Set B (cVAE)"),
    ("#f39c12", "-",  "350 + Set C (cVAE)"),
]
for (name, (acc, hist, _)), (color, ls, label) in zip(results.items(), curve_styles):
    axes[1].plot(epoch_x, hist, color=color, linestyle=ls, lw=1.8, label=label)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy (%)")
axes[1].set_title("Learning Curves — All Configurations", fontweight="bold")
axes[1].legend(fontsize=8.5)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(70, 100)

plt.tight_layout()
plt.savefig("step4_accuracy_comparison.png", dpi=130, bbox_inches="tight")
plt.show()
print("\n  [✓] Saved → step4_accuracy_comparison.png")

# ─────────────────────────────────────────────
# PLOT 2: Per-class accuracy for best config vs baseline
# ─────────────────────────────────────────────
def per_class_acc(model_path_or_model, loader):
    if isinstance(model_path_or_model, str):
        m = LeNet5().to(DEVICE)
        m.load_state_dict(torch.load(model_path_or_model))
    else:
        m = model_path_or_model
    m.eval()
    correct = [0]*NUM_CLASSES
    total   = [0]*NUM_CLASSES
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE) if isinstance(labels, torch.Tensor) \
                     else torch.tensor(labels).to(DEVICE)
            preds  = m(imgs).argmax(1)
            for d in range(NUM_CLASSES):
                mask = (labels == d)
                correct[d] += (preds[mask] == d).sum().item()
                total[d]   += mask.sum().item()
    return [100*correct[d]/total[d] if total[d]>0 else 0 for d in range(NUM_CLASSES)]

# Re-train baseline and best cVAE model to get per-class accuracy
print("\n  Computing per-class accuracy for baseline vs best cVAE set ...")

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
_, _, model_base = train_lenet(make_loader(real350_ds), test_loader,
                                "Baseline-350 [per-class eval]")

# Find best cVAE configuration
cvae_configs = {k: v for k, v in results.items() if "Set" in k}
best_name    = max(cvae_configs, key=lambda k: cvae_configs[k][0])
best_ds_map  = {
    "350 real\n+ Set A": combined_A,
    "350 real\n+ Set B": combined_B,
    "350 real\n+ Set C": combined_C,
}
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
_, _, model_best = train_lenet(make_loader(best_ds_map[best_name]), test_loader,
                                f"Best cVAE [{best_name.replace(chr(10),' ')}] [per-class eval]")

pc_base = per_class_acc(model_base, test_loader)
pc_best = per_class_acc(model_best, test_loader)

x = np.arange(NUM_CLASSES)
w = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, pc_base, w, label="Baseline-350", color="#95a5a6", edgecolor="white")
ax.bar(x + w/2, pc_best, w,
       label=f"Best cVAE: {best_name.replace(chr(10),' ')}",
       color="#e74c3c", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(range(NUM_CLASSES))
ax.set_xlabel("Digit Class")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-Class Accuracy: Baseline-350 vs Best cVAE Config", fontweight="bold")
ax.legend()
ax.set_ylim(70, 101)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("step4_perclass_accuracy.png", dpi=130, bbox_inches="tight")
plt.show()
print("  [✓] Saved → step4_perclass_accuracy.png")

# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY — cVAE (Problem 1)")
print("=" * 65)
print(f"  {'Configuration':<30} | {'Train Size':>10} | {'Test Acc':>10} | {'Δ vs 350':>10}")
print(f"  {'─'*63}")
for name, (acc, _, size) in results.items():
    label = name.replace("\n", " ")
    delta = acc - baseline_acc
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<30} | {size:>10,} | {acc:>9.2f}% | {sign}{delta:.2f}%")
print(f"  {'─'*63}")
print(f"\n  Best cVAE strategy  : {best_name.replace(chr(10), ' ')}")
print(f"  Best cVAE accuracy  : {cvae_configs[best_name][0]:.2f}%")
best_cvae_acc = cvae_configs[best_name][0]
real1000_acc  = results["Baseline\n1000 real"][0]
gap_closed    = (best_cvae_acc - baseline_acc) / (real1000_acc - baseline_acc) * 100
print(f"  Gap to 1000-real    : {real1000_acc - best_cvae_acc:.2f}%")
print(f"  Gap closed by cVAE  : {gap_closed:.1f}% of the 350→1000 gap")
print("=" * 65)
print("\n✅  Step 4 complete — ready for final report (Step 5).")