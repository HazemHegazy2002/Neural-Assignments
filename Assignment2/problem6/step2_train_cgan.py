import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
RESULTS_DIR = "./results"
MODELS_DIR  = "./results/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────
NOISE_DIM    = 100    # size of random noise vector
EMBED_DIM    = 50     # size of label embedding
NUM_CLASSES  = 10     # digits 0-9
IMAGE_SIZE   = 28     # 28x28 pixels
CHANNELS     = 1      # grayscale
BATCH_SIZE   = 64     # images per batch
EPOCHS       = 300    # increased from 100 to 300
LR_G         = 0.0002 # Generator learning rate
LR_D         = 0.0001 # Discriminator learning rate (slower to balance)
BETA1        = 0.5    # Adam beta1 (from DCGAN paper)
G_UPDATES    = 2      # Train Generator twice per Discriminator update

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("\nLoading ReducedMNIST...")

train_images = torch.load(f"{RESULTS_DIR}/train_images.pt")
train_labels = torch.load(f"{RESULTS_DIR}/train_labels.pt")

print(f"Train images shape : {train_images.shape}")
print(f"Train labels shape : {train_labels.shape}")

# Create DataLoader
dataset    = TensorDataset(train_images, train_labels)
dataloader = DataLoader(
    dataset,
    batch_size = BATCH_SIZE,
    shuffle    = True
)
print(f"Number of batches  : {len(dataloader)}")

# ─────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Label embedding
        self.label_embed = nn.Embedding(NUM_CLASSES, EMBED_DIM)

        # First layer
        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM + EMBED_DIM, 7 * 7 * 256),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.ReLU()
        )

        # Convolutional layers
        self.conv_layers = nn.Sequential(

            # 7x7x256 → 14x14x128
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 14x14x128 → 28x28x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 28x28x64 → 28x28x1
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
# DISCRIMINATOR
# ─────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Label embedding
        self.label_embed = nn.Embedding(NUM_CLASSES, IMAGE_SIZE * IMAGE_SIZE)

        # Convolutional layers
        self.conv_layers = nn.Sequential(

            # 28x28x2 → 14x14x64
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),      # added dropout to weaken discriminator

            # 14x14x64 → 7x7x128
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),      # added dropout to weaken discriminator

            # 7x7x128 → 4x4x256
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # Final classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        label_map = self.label_embed(labels)
        label_map = label_map.view(-1, 1, 28, 28)
        x = torch.cat([images, label_map], dim=1)
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

# ─────────────────────────────────────────
# BUILD MODELS
# ─────────────────────────────────────────
generator     = Generator().to(device)
discriminator = Discriminator().to(device)

print("\n" + "="*50)
print("GENERATOR ARCHITECTURE")
print("="*50)
print(generator)

print("\n" + "="*50)
print("DISCRIMINATOR ARCHITECTURE")
print("="*50)
print(discriminator)

gen_params  = sum(p.numel() for p in generator.parameters())
disc_params = sum(p.numel() for p in discriminator.parameters())
print(f"\nGenerator parameters     : {gen_params:,}")
print(f"Discriminator parameters : {disc_params:,}")

# ─────────────────────────────────────────
# LOSS & OPTIMIZERS
# ─────────────────────────────────────────
criterion = nn.BCELoss()

# Different learning rates for G and D
optimizer_G = optim.Adam(
    generator.parameters(),
    lr    = LR_G,
    betas = (BETA1, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr    = LR_D,      # slower learning rate
    betas = (BETA1, 0.999)
)

# ─────────────────────────────────────────
# FIXED NOISE FOR VISUALIZATION
# ─────────────────────────────────────────
fixed_noise  = torch.randn(10, NOISE_DIM).to(device)
fixed_labels = torch.arange(0, 10).to(device)

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING CGAN")
print("="*50)
print(f"Epochs          : {EPOCHS}")
print(f"Generator LR    : {LR_G}")
print(f"Discriminator LR: {LR_D}")
print(f"G updates/batch : {G_UPDATES}")
print("="*50)

G_losses = []
D_losses = []
img_list = []

training_start = time.time()

for epoch in range(EPOCHS):
    epoch_G_loss = 0
    epoch_D_loss = 0

    for batch_imgs, batch_labels in dataloader:
        batch_imgs   = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        batch_size   = batch_imgs.size(0)

        real_targets = torch.ones(batch_size, 1).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        # ─────────────────────────
        # TRAIN DISCRIMINATOR ONCE
        # ─────────────────────────
        optimizer_D.zero_grad()

        # Real images loss
        real_output = discriminator(batch_imgs, batch_labels)
        loss_real   = criterion(real_output, real_targets)

        # Fake images loss
        noise     = torch.randn(batch_size, NOISE_DIM).to(device)
        fake_imgs = generator(noise, batch_labels)
        fake_output = discriminator(fake_imgs.detach(), batch_labels)
        loss_fake   = criterion(fake_output, fake_targets)

        # Update Discriminator
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()
        epoch_D_loss += loss_D.item()

        # ─────────────────────────
        # TRAIN GENERATOR TWICE
        # ─────────────────────────
        g_loss_total = 0
        for _ in range(G_UPDATES):
            optimizer_G.zero_grad()

            # Generate new fake images
            noise     = torch.randn(batch_size, NOISE_DIM).to(device)
            fake_imgs = generator(noise, batch_labels)

            # Generator wants Discriminator to say REAL
            fake_output2 = discriminator(fake_imgs, batch_labels)
            loss_G       = criterion(fake_output2, real_targets)

            loss_G.backward()
            optimizer_G.step()
            g_loss_total += loss_G.item()

        epoch_G_loss += g_loss_total / G_UPDATES

    # Average losses per epoch
    avg_G = epoch_G_loss / len(dataloader)
    avg_D = epoch_D_loss / len(dataloader)
    G_losses.append(avg_G)
    D_losses.append(avg_D)

    # Print every 30 epochs
    if (epoch + 1) % 30 == 0:
        elapsed = (time.time() - training_start) / 60
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"G Loss: {avg_G:.4f}  "
              f"D Loss: {avg_D:.4f}  "
              f"Time: {elapsed:.1f} min")

        # Save sample images
        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise, fixed_labels)
            img_list.append(samples.cpu())
        generator.train()

training_end  = time.time()
training_time = (training_end - training_start) * 1000

print(f"\nTotal training time: {training_time:.1f} ms")
print(f"Total training time: {training_time/1000/60:.1f} minutes")

# ─────────────────────────────────────────
# PLOT LOSSES
# ─────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(G_losses, label='Generator Loss',     color='blue')
plt.plot(D_losses, label='Discriminator Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CGAN Training Losses (300 Epochs)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/cgan_loss_curve.png")
plt.show()
print("Loss curve saved!")

# ─────────────────────────────────────────
# VISUALIZE TRAINING PROGRESS
# ─────────────────────────────────────────
print("\nVisualizing training progress...")

fig, axes = plt.subplots(
    len(img_list), 10,
    figsize=(20, 2 * len(img_list))
)

for i, imgs in enumerate(img_list):
    for j in range(10):
        ax  = axes[i, j] if len(img_list) > 1 else axes[j]
        img = imgs[j].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(f'Digit {j}')
    axes[i, 0].set_ylabel(
        f'Epoch {(i+1)*30}',
        rotation = 0,
        labelpad = 40
    )

plt.suptitle('Generated Images Progress Per Epoch', fontsize=14)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_progress.png")
plt.show()
print("Training progress saved!")

# ─────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────
print("\nSaving models...")
torch.save(generator.state_dict(),
           f"{MODELS_DIR}/generator.pth")
torch.save(discriminator.state_dict(),
           f"{MODELS_DIR}/discriminator.pth")

print(f"Saved:")
print(f"  generator.pth     → trained generator")
print(f"  discriminator.pth → trained discriminator")

print("\n" + "="*50)
print("STEP 2 SUMMARY")
print("="*50)
print(f"Epochs trained    : {EPOCHS}")
print(f"G updates/batch   : {G_UPDATES}")
print(f"Final G Loss      : {G_losses[-1]:.4f}")
print(f"Final D Loss      : {D_losses[-1]:.4f}")
print(f"Training time     : {training_time:.1f} ms")
print(f"Training time     : {training_time/1000/60:.1f} minutes")

print("\n✅ Step 2 Complete — Improved CGAN trained successfully!")