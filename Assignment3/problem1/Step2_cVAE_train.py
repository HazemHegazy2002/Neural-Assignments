"""
Problem 1 - Step 2 (cVAE): Conditional VAE Training
=====================================================


Key differences vs cDCGAN:
  - Encoder  : image + label → (mu, logvar)  in latent space
  - Decoder  : z + label     → reconstructed image
  - Loss     : ELBO = Reconstruction (BCE) + KL Divergence
  - No adversarial training, no G/D balance tricks needed
  - Sampling at inference: z ~ N(0,I), conditioned on class label
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
print("  Step 2 — Conditional VAE Training  (GPU + RAM Preload)")
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
IMG_SIZE       = 28
CHANNELS       = 1
LATENT_DIM     = 128        # cVAE latent dimension (larger than GAN z helps quality)
EMBED_DIM      = 50         # class embedding size
AUGMENT_FACTOR = 15

BATCH_SIZE     = 128
NUM_EPOCHS     = 100
LR             = 2e-4       # single learning rate (one optimizer)
BETA1, BETA2   = 0.9, 0.999
# KL weight: start low (focus on reconstruction), anneal up to 1.0
# This avoids posterior collapse in early training.
KL_WEIGHT_MAX  = 1.0
SEED           = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

print(f"\n  Batch size  : {BATCH_SIZE}  |  Epochs    : {NUM_EPOCHS}")
print(f"  Latent dim  : {LATENT_DIM}  |  Embed dim : {EMBED_DIM}")
print(f"  LR          : {LR}  (single Adam optimizer)")
print(f"  KL annealing: 0 → {KL_WEIGHT_MAX} over first 50 epochs")

# ─────────────────────────────────────────────
# RAM-PRELOADED DATASET  (unchanged from cDCGAN)
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
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
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
    if "Conv" in cn:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cn:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif "Linear" in cn:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):
    """
    Encodes (image, label) → (mu, logvar) in latent space.
    Input : [B, 1, 32, 32] image  +  [B] label
    Output: mu [B, LATENT_DIM],  logvar [B, LATENT_DIM]
    """
    def __init__(self):
        super().__init__()
        # Label is embedded and projected to a spatial map concatenated with image
        self.label_embed = nn.Embedding(NUM_CLASSES, IMG_SIZE * IMG_SIZE)

        # Conv stack: input is 2-channel (image + label map)
        self.conv = nn.Sequential(
            nn.Conv2d(2,    64,  4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),          # → 16×16
            nn.Conv2d(64,  128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),  # → 8×8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),  # → 4×4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),  # → 2×2
        )
        flat_dim = 512 * 2 * 2  # = 2048

        self.fc_mu     = nn.Linear(flat_dim, LATENT_DIM)
        self.fc_logvar = nn.Linear(flat_dim, LATENT_DIM)

    def forward(self, img, labels):
        # Concatenate spatial label map with image
        e   = self.label_embed(labels).view(-1, 1, IMG_SIZE, IMG_SIZE)
        x   = self.conv(torch.cat([img, e], dim=1))   # [B, 512, 2, 2]
        x   = x.view(x.size(0), -1)                   # [B, 2048]
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decodes (z, label) → reconstructed image.
    Input : z [B, LATENT_DIM]  +  label [B]
    Output: image [B, 1, 32, 32]  (Tanh output, range [-1, 1])
    """
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(NUM_CLASSES, EMBED_DIM)

        # Project concatenated (z, label_embed) to spatial feature map
        self.project = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 512 * 4 * 4),
            nn.ReLU(True),
        )

        # Transposed conv stack: 4×4 → 8 → 16 → 32
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, CHANNELS, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        e = self.label_embed(labels)                               # [B, EMBED_DIM]
        x = self.project(torch.cat([z, e], dim=1))                # [B, 512*4*4]
        x = x.view(-1, 512, 4, 4)                                 # [B, 512, 4, 4]
        return self.conv(x)                                        # [B, 1, 32, 32]


# ─────────────────────────────────────────────
# REPARAMETERISATION TRICK
# ─────────────────────────────────────────────
def reparameterise(mu, logvar):
    """
    Sample z ~ N(mu, sigma^2) using the reparameterisation trick:
        z = mu + eps * sigma,   eps ~ N(0, I)
    Keeps gradients flowing through mu and logvar.
    """
    std = torch.exp(0.5 * logvar)       # sigma = exp(logvar / 2)
    eps = torch.randn_like(std)         # eps ~ N(0, I)
    return mu + eps * std


# ─────────────────────────────────────────────
# ELBO LOSS
# ─────────────────────────────────────────────
def elbo_loss(recon, target, mu, logvar, kl_weight=1.0):
    """
    ELBO = Reconstruction loss + β * KL divergence

    Reconstruction : MSE loss (works well for normalised grayscale images)
                     Alternatively use BCE if images are in [0,1].
    KL divergence  : -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                     Closed form for N(mu, sigma) vs N(0, I).

    kl_weight (β)  : KL annealing weight; start low to avoid posterior collapse.
    """
    # MSE reconstruction loss (per element, then mean over batch)
    recon_loss = F.mse_loss(recon, target, reduction="mean")

    # KL divergence (normalised by batch size × latent dim for stability)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_weight * kl, recon_loss.item(), kl.item()


# ─────────────────────────────────────────────
# INSTANTIATE MODELS
# ─────────────────────────────────────────────
encoder = Encoder().to(DEVICE);  encoder.apply(weights_init)
decoder = Decoder().to(DEVICE);  decoder.apply(weights_init)

print(f"\n  Encoder params : {sum(p.numel() for p in encoder.parameters()):,}")
print(f"  Decoder params : {sum(p.numel() for p in decoder.parameters()):,}")
print("  ✅ Both models on GPU")

# ─────────────────────────────────────────────
# OPTIMISER & AMP
# ─────────────────────────────────────────────
# Single optimiser over all parameters (encoder + decoder)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=LR, betas=(BETA1, BETA2))
scaler    = torch.amp.GradScaler()

# Fixed latent vectors for visualisation (one z per class × 10 samples)
fixed_z      = torch.randn(NUM_CLASSES * 10, LATENT_DIM, device=DEVICE)
fixed_labels = torch.tensor([d for d in range(NUM_CLASSES) for _ in range(10)],
                             device=DEVICE)

# ─────────────────────────────────────────────
# KL ANNEALING SCHEDULE
# ─────────────────────────────────────────────
ANNEAL_EPOCHS = 50   # linearly increase KL weight from 0 → KL_WEIGHT_MAX

def get_kl_weight(epoch):
    """Linear warmup for KL weight to prevent posterior collapse."""
    return min(KL_WEIGHT_MAX, KL_WEIGHT_MAX * epoch / ANNEAL_EPOCHS)

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
total_losses  = []
recon_losses  = []
kl_losses     = []

print(f"\n{'Epoch':>6} | {'Total':>8} | {'Recon':>8} | {'KL':>8} | "
      f"{'β(KL)':>6} | {'VRAM MB':>8} | {'Time s':>7}")
print("-" * 72)

total_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    kl_weight = get_kl_weight(epoch)

    acc_total = acc_recon = acc_kl = 0.0

    encoder.train(); decoder.train()

    for real_imgs, labels in loader:
        real_imgs = real_imgs.to(DEVICE, non_blocking=True)
        labels    = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            # ── Forward pass ──────────────────────────────────────
            mu, logvar = encoder(real_imgs, labels)       # encode
            z          = reparameterise(mu, logvar)       # sample latent
            recon_imgs = decoder(z, labels)               # decode

            # ── ELBO loss ─────────────────────────────────────────
            loss, r_val, kl_val = elbo_loss(recon_imgs, real_imgs,
                                            mu, logvar, kl_weight)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        acc_total += loss.item()
        acc_recon += r_val
        acc_kl    += kl_val

    n          = len(loader)
    avg_total  = acc_total / n
    avg_recon  = acc_recon / n
    avg_kl     = acc_kl    / n

    total_losses.append(avg_total)
    recon_losses.append(avg_recon)
    kl_losses.append(avg_kl)

    vram_mb = torch.cuda.memory_allocated() / 1e6
    elapsed = time.time() - t0

    print(f"{epoch:>6} | {avg_total:>8.4f} | {avg_recon:>8.4f} | "
          f"{avg_kl:>8.4f} | {kl_weight:>6.3f} | {vram_mb:>8.1f} | {elapsed:>6.1f}s")

total_min = (time.time() - total_start) / 60
print(f"\n  Total training time : {total_min:.1f} min")

# ─────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")
print("[✓] Saved → encoder.pth  &  decoder.pth")

# ─────────────────────────────────────────────
# LOSS CURVES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(total_losses, color="steelblue", lw=1.5)
axes[0].set_title("Total ELBO Loss");  axes[0].set_xlabel("Epoch")
axes[0].grid(alpha=0.3)

axes[1].plot(recon_losses, color="tomato", lw=1.5)
axes[1].set_title("Reconstruction Loss (MSE)"); axes[1].set_xlabel("Epoch")
axes[1].grid(alpha=0.3)

axes[2].plot(kl_losses, color="seagreen", lw=1.5)
axes[2].set_title("KL Divergence"); axes[2].set_xlabel("Epoch")
axes[2].grid(alpha=0.3)

plt.suptitle("cVAE Training Losses", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("step2_cvae_loss_curves.png", dpi=120)
plt.show()
print("[✓] Saved → step2_cvae_loss_curves.png")

# ─────────────────────────────────────────────
# GENERATED SAMPLE GRID
# ─────────────────────────────────────────────
encoder.eval(); decoder.eval()

with torch.no_grad():
    gen_imgs = decoder(fixed_z, fixed_labels).cpu()

gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)

fig, axes = plt.subplots(10, NUM_CLASSES, figsize=(12, 12))
fig.suptitle("cVAE Generated Samples — End of Training\n(cols=0–9, rows=10 samples)",
             fontsize=12, fontweight="bold")
for i in range(NUM_CLASSES):
    axes[0, i].set_title(str(i), fontsize=12, fontweight="bold", pad=10)
    for j in range(10):
        axes[j, i].imshow(gen_imgs[i * 10 + j].squeeze(), cmap="gray")
        axes[j, i].axis("off")
plt.tight_layout()
plt.savefig("step2_cvae_generated_samples.png", dpi=120, bbox_inches="tight")
plt.show()
print("[✓] Saved → step2_cvae_generated_samples.png")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  TRAINING SUMMARY")
print("=" * 62)
print(f"  GPU              : {torch.cuda.get_device_name(0)}")
print(f"  Epochs           : {NUM_EPOCHS}")
print(f"  Final Total Loss : {total_losses[-1]:.4f}")
print(f"  Final Recon Loss : {recon_losses[-1]:.4f}  (target: < 0.05)")
print(f"  Final KL Loss    : {kl_losses[-1]:.4f}   (target: 1 – 10)")
ok_recon = recon_losses[-1] < 0.10
ok_kl    = 0.5 <= kl_losses[-1] <= 20.0
print(f"  Reconstruction   : {'✅ GOOD' if ok_recon else '⚠️  High — consider more epochs'}")
print(f"  KL               : {'✅ GOOD' if ok_kl    else '⚠️  Posterior collapse or explosion'}")
print(f"  Total time       : {total_min:.1f} min")
print(f"  Peak VRAM        : {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
print("=" * 62)
print("\n✅  Step 2 complete — ready for sample generation & filtering (Step 3).")