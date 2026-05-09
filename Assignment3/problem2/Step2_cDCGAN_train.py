"""
Problem 2 - Step 2 (v2): cDCGAN Training — Balanced Training Fix
=================================================================
Fixes applied vs v1:
  1. G trained TWICE per batch  → prevents D from dominating
  2. D lr = 0.0001, G lr = 0.0002  → G gets slightly more update power
  3. Instance noise on D inputs  → slows D from memorising
  4. Gradient clipping on G  → prevents G loss from exploding
  5. All images preloaded in RAM  → no disk I/O bottleneck
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# GPU VALIDATION
# ─────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. Install CUDA PyTorch first.")

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print("=" * 62)
print("  Step 2 v2 — cDCGAN Balanced Training  (GPU + RAM Preload)")
print("=" * 62)
print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"  CUDA        : {torch.version.cuda}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT      = "/kaggle/input/datasets/mohamedgamal07/reduced-mnist/Reduced MNIST Data"
TRAIN_DIR      = os.path.join(DATA_ROOT, "Reduced Trainging data")
NUM_CLASSES    = 10
IMG_SIZE       = 32
CHANNELS       = 1
LATENT_DIM     = 100
EMBED_DIM      = 50
AUGMENT_FACTOR = 15

BATCH_SIZE     = 128
NUM_EPOCHS     = 100
LR_G           = 0.0002   # G gets more learning rate
LR_D           = 0.0001   # D gets less  → prevents D from crushing G
BETA1, BETA2   = 0.5, 0.999
NOISE_STD_INIT = 0.1      # instance noise added to D inputs (decays to 0)
SEED           = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

print(f"\n  Batch size  : {BATCH_SIZE}  |  Epochs : {NUM_EPOCHS}")
print(f"  LR_G={LR_G}  LR_D={LR_D}  (asymmetric to balance training)")
print(f"  G trained 2× per batch  |  Instance noise: {NOISE_STD_INIT} → 0")

# ─────────────────────────────────────────────
# RAM-PRELOADED DATASET
# ─────────────────────────────────────────────
class PreloadedAugDataset(Dataset):
    def __init__(self, root, max_per_class=350, augment_factor=AUGMENT_FACTOR):
        self.augment_factor = augment_factor
        self.images, self.labels = [], []
        print(f"\n  Pre-loading images into RAM ...")
        resize = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ])
        for digit in range(NUM_CLASSES):
            folder = os.path.join(root, str(digit))
            files  = sorted([f for f in os.listdir(folder)
                             if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))])
            for fname in files[:max_per_class]:
                img = Image.open(os.path.join(folder, fname)).convert("L")
                self.images.append(resize(img))
                self.labels.append(digit)
        print(f"  Loaded {len(self.images)} images into RAM  ✓")

    def __len__(self):
        return len(self.images) * (1 + self.augment_factor)

    def __getitem__(self, idx):
        oi  = idx // (1 + self.augment_factor)
        ac  = idx  % (1 + self.augment_factor)
        img = self.images[oi]
        lbl = self.labels[oi]
        if ac == 0:
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        else:
            tf = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.85,1.15)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        return tf(img), lbl

dataset = PreloadedAugDataset(TRAIN_DIR, max_per_class=350)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=0, pin_memory=True, drop_last=True)
print(f"  Dataset : {len(dataset):,}  |  {len(loader)} batches/epoch")

# ─────────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────────
def weights_init(m):
    cn = m.__class__.__name__
    if "Conv" in cn:    nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cn:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
            nn.Conv2d(64, CHANNELS, 3, 1, 1, bias=False), nn.Tanh())
    def forward(self, z, labels):
        e = self.label_embed(labels)
        return self.conv(self.project(torch.cat([z,e],1)).view(-1,512,4,4))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(NUM_CLASSES, IMG_SIZE*IMG_SIZE)
        self.conv = nn.Sequential(
            nn.Conv2d(2,   64,  4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,  128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512*2*2, 1))
    def forward(self, img, labels):
        e = self.label_embed(labels).view(-1, 1, IMG_SIZE, IMG_SIZE)
        return self.fc(self.conv(torch.cat([img, e], 1)))

G = Generator().to(DEVICE);       G.apply(weights_init)
D = Discriminator().to(DEVICE);   D.apply(weights_init)
print(f"\n  Generator params     : {sum(p.numel() for p in G.parameters()):,}")
print(f"  Discriminator params : {sum(p.numel() for p in D.parameters()):,}")
print("  ✅ Both models on GPU")

# ─────────────────────────────────────────────
# LOSS / OPTIMISERS / AMP
# ─────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
opt_G  = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
opt_D  = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))
scaler = torch.amp.GradScaler()

fixed_z      = torch.randn(NUM_CLASSES*10, LATENT_DIM, device=DEVICE)
fixed_labels = torch.tensor([d for d in range(NUM_CLASSES) for _ in range(10)], device=DEVICE)

def real_lbl(n): return torch.full((n,1), 0.9,  device=DEVICE)
def fake_lbl(n): return torch.full((n,1), 0.05, device=DEVICE)

def instance_noise(imgs, std):
    """Add small Gaussian noise to D inputs — regularises D."""
    if std <= 0: return imgs
    return imgs + torch.randn_like(imgs) * std

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
G_losses, D_losses = [], []

print(f"\n{'Epoch':>6} | {'D Loss':>8} | {'G Loss':>8} | {'VRAM MB':>8} | {'Time s':>7}")
print("-" * 55)

total_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    g_acc = d_acc = 0.0

    # Instance noise std decays linearly to 0 over all epochs
    noise_std = NOISE_STD_INIT * max(0.0, 1.0 - epoch / NUM_EPOCHS)

    G.train(); D.train()
    for real_imgs, labels in loader:
        real_imgs = real_imgs.to(DEVICE, non_blocking=True)
        labels    = labels.to(DEVICE, non_blocking=True)
        bsz       = real_imgs.size(0)

        # ── Train Discriminator ──────────────────────────────
        opt_D.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            noisy_real = instance_noise(real_imgs, noise_std)
            z          = torch.randn(bsz, LATENT_DIM, device=DEVICE)
            fake_imgs  = G(z, labels).detach()
            noisy_fake = instance_noise(fake_imgs, noise_std)
            loss_D = 0.5 * (criterion(D(noisy_real, labels), real_lbl(bsz)) +
                            criterion(D(noisy_fake, labels), fake_lbl(bsz)))
        scaler.scale(loss_D).backward()
        scaler.step(opt_D)

        # ── Train Generator TWICE per batch ─────────────────
        g_loss_sum = 0.0
        for _ in range(2):
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                z      = torch.randn(bsz, LATENT_DIM, device=DEVICE)
                loss_G = criterion(D(G(z, labels), labels), real_lbl(bsz))
            scaler.scale(loss_G).backward()
            # Gradient clipping — prevents G loss from exploding
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            scaler.step(opt_G)
            scaler.update()
            g_loss_sum += loss_G.item()

        d_acc += loss_D.item()
        g_acc += g_loss_sum / 2   # average of 2 G updates

    d_avg = d_acc / len(loader)
    g_avg = g_acc / len(loader)
    G_losses.append(g_avg)
    D_losses.append(d_avg)
    vram_mb = torch.cuda.memory_allocated() / 1e6
    elapsed = time.time() - t0

    # Print every epoch so you can monitor balance
    print(f"{epoch:>6} | {d_avg:>8.4f} | {g_avg:>8.4f} | {vram_mb:>8.1f} | {elapsed:>6.1f}s")

total_min = (time.time() - total_start) / 60
print(f"\n  Total training time : {total_min:.1f} min")

# ─────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")
print("[✓] Saved → generator.pth  &  discriminator.pth")

# ─────────────────────────────────────────────
# LOSS CURVES
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(G_losses, label="Generator",     color="steelblue", lw=1.5)
plt.plot(D_losses, label="Discriminator", color="tomato",    lw=1.5)
plt.axhline(y=0.693, color="gray", linestyle="--", alpha=0.6, label="Ideal D Loss (ln2≈0.693)")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("cDCGAN Training Loss (Balanced)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("step2_loss_curves.png", dpi=120)
plt.show()
print("[✓] Saved → step2_loss_curves.png")

# ─────────────────────────────────────────────
# GENERATED SAMPLE GRID
# ─────────────────────────────────────────────
G.eval()
with torch.no_grad():
    gen_imgs = G(fixed_z, fixed_labels).cpu()
gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)

fig, axes = plt.subplots(10, NUM_CLASSES, figsize=(12, 12))
fig.suptitle("cDCGAN Generated Samples — End of Training\n(cols=0–9, rows=10 samples)",
             fontsize=12, fontweight="bold")
for i in range(NUM_CLASSES):
    axes[0, i].set_title(str(i), fontsize=12, fontweight="bold", pad=10)
    for j in range(10):
        axes[j, i].imshow(gen_imgs[i*10+j].squeeze(), cmap="gray")
        axes[j, i].axis("off")
plt.tight_layout()
plt.savefig("step2_generated_samples.png", dpi=120, bbox_inches="tight")
plt.show()
print("[✓] Saved → step2_generated_samples.png")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  TRAINING SUMMARY")
print("=" * 62)
print(f"  GPU              : {torch.cuda.get_device_name(0)}")
print(f"  Epochs           : {NUM_EPOCHS}")
print(f"  Final G Loss     : {G_losses[-1]:.4f}   (target: 0.8 – 1.8)")
print(f"  Final D Loss     : {D_losses[-1]:.4f}   (target: 0.4 – 0.7)")
balanced = 0.4 <= D_losses[-1] <= 0.75 and G_losses[-1] <= 2.2
print(f"  Balance check    : {'✅ GOOD' if balanced else '⚠️  Still imbalanced — check sample grid'}")
print(f"  Total time       : {total_min:.1f} min")
print(f"  Peak VRAM        : {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
print("=" * 62)
print("\n✅  Step 2 complete — ready for sample generation & filtering (Step 3).")