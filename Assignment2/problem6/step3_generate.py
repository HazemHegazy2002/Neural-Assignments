import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
RESULTS_DIR = "./results"
MODELS_DIR  = "./results/models"
GEN_DIR     = "./results/generated_images"
os.makedirs(GEN_DIR, exist_ok=True)

# ─────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────
NOISE_DIM   = 100
EMBED_DIM   = 50
NUM_CLASSES = 10
IMAGE_SIZE  = 28
CHANNELS    = 1
SAMPLES     = 3   # 3 images per digit as required

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
# GENERATOR ARCHITECTURE
# Must be same as step2
# ─────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_embed = nn.Embedding(NUM_CLASSES, EMBED_DIM)

        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM + EMBED_DIM, 7 * 7 * 256),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, CHANNELS, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_vec = self.label_embed(labels)
        x = torch.cat([noise, label_vec], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv_layers(x)
        return x

# ─────────────────────────────────────────
# LOAD TRAINED GENERATOR
# ─────────────────────────────────────────
print("\nLoading trained generator...")

generator = Generator().to(device)
generator.load_state_dict(
    torch.load(f"{MODELS_DIR}/generator.pth",
               map_location=device)
)
generator.eval()  # set to evaluation mode
print("Generator loaded successfully!")

# ─────────────────────────────────────────
# LOAD REAL IMAGES FOR COMPARISON
# ─────────────────────────────────────────
print("Loading real images for comparison...")

train_images = torch.load(f"{RESULTS_DIR}/train_images.pt")
train_labels = torch.load(f"{RESULTS_DIR}/train_labels.pt")

# ─────────────────────────────────────────
# GENERATE 3 IMAGES PER DIGIT
# ─────────────────────────────────────────
print(f"\nGenerating {SAMPLES} images per digit...")

all_generated = {}  # digit → list of generated images

generation_start = time.time()

with torch.no_grad():
    for digit in range(NUM_CLASSES):
        # Create 3 different random noises
        noise  = torch.randn(SAMPLES, NOISE_DIM).to(device)

        # Create label tensor [digit, digit, digit]
        labels = torch.full(
            (SAMPLES,), digit,
            dtype=torch.long
        ).to(device)

        # Generate 3 images of this digit
        generated = generator(noise, labels)

        # Move to CPU and store
        all_generated[digit] = generated.cpu()

        print(f"  Digit {digit}: {SAMPLES} images generated ✅")

generation_end  = time.time()
generation_time = (generation_end - generation_start) * 1000

print(f"\nGeneration time: {generation_time:.1f} ms")

# ─────────────────────────────────────────
# MAIN VISUALIZATION
# Real vs Generated side by side
# ─────────────────────────────────────────
print("\nCreating main visualization...")

# Layout:
# Row 1: Real images
# Row 2: Generated sample 1
# Row 3: Generated sample 2
# Row 4: Generated sample 3

fig, axes = plt.subplots(
    4, NUM_CLASSES,
    figsize=(20, 8)
)

for digit in range(NUM_CLASSES):

    # ── Row 1: Real image ──
    real_idx = (train_labels == digit).nonzero(as_tuple=True)[0][0]
    real_img  = train_images[real_idx].squeeze().numpy()
    axes[0, digit].imshow(real_img, cmap='gray')
    axes[0, digit].set_title(f'Digit {digit}', fontsize=10)
    axes[0, digit].axis('off')

    # ── Rows 2-4: Generated images ──
    for sample in range(SAMPLES):
        gen_img = all_generated[digit][sample].squeeze().numpy()

        # Normalize from [-1,1] to [0,1] for display
        gen_img = (gen_img + 1) / 2

        axes[sample + 1, digit].imshow(gen_img, cmap='gray')
        axes[sample + 1, digit].axis('off')

# Row labels
axes[0, 0].set_ylabel('Real',      rotation=0, labelpad=40, fontsize=10)
axes[1, 0].set_ylabel('Generated\nSample 1', rotation=0, labelpad=50, fontsize=9)
axes[2, 0].set_ylabel('Generated\nSample 2', rotation=0, labelpad=50, fontsize=9)
axes[3, 0].set_ylabel('Generated\nSample 3', rotation=0, labelpad=50, fontsize=9)

plt.suptitle(
    'Real vs GAN Generated Images\n(Trained on 350 examples per digit)',
    fontsize=14
)
plt.tight_layout()
plt.savefig(f"{GEN_DIR}/generated_3_per_digit.png", dpi=150)
plt.show()
print("Main visualization saved!")

# ─────────────────────────────────────────
# SAVE INDIVIDUAL IMAGES
# ─────────────────────────────────────────
print("\nSaving individual images...")

for digit in range(NUM_CLASSES):
    for sample in range(SAMPLES):
        img = all_generated[digit][sample].squeeze().numpy()
        img = (img + 1) / 2  # normalize to [0,1]

        plt.figure(figsize=(2, 2))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'Digit {digit} - Sample {sample+1}')
        plt.tight_layout()
        plt.savefig(
            f"{GEN_DIR}/digit_{digit}_sample_{sample+1}.png",
            dpi=100
        )
        plt.close()

print(f"Saved {NUM_CLASSES * SAMPLES} individual images!")

# ─────────────────────────────────────────
# QUALITY ANALYSIS
# Compare generated vs real pixel statistics
# ─────────────────────────────────────────
print("\n" + "="*50)
print("QUALITY ANALYSIS")
print("="*50)

for digit in range(NUM_CLASSES):
    # Get real images of this digit
    real_idx  = (train_labels == digit).nonzero(as_tuple=True)[0]
    real_imgs = train_images[real_idx].numpy()

    # Get generated images of this digit
    gen_imgs  = all_generated[digit].numpy()
    gen_imgs  = (gen_imgs + 1) / 2  # normalize to [0,1]

    # Calculate statistics
    real_mean = real_imgs.mean()
    real_std  = real_imgs.std()
    gen_mean  = gen_imgs.mean()
    gen_std   = gen_imgs.std()

    # Pixel difference
    difference = abs(real_mean - gen_mean)

    print(f"Digit {digit}:")
    print(f"  Real      → mean: {real_mean:.3f}  std: {real_std:.3f}")
    print(f"  Generated → mean: {gen_mean:.3f}  std: {gen_std:.3f}")
    print(f"  Difference: {difference:.3f}")

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*50)
print("STEP 3 SUMMARY")
print("="*50)
print(f"Images generated  : {NUM_CLASSES * SAMPLES} total")
print(f"  Per digit       : {SAMPLES}")
print(f"  Digits covered  : 0-9")
print(f"Generation time   : {generation_time:.1f} ms")
print(f"Saved to          : {GEN_DIR}/")

print("\n✅ Step 3 Complete — Images generated successfully!")