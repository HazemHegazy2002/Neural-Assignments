import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


RESULTS_DIR = "./results"


class LinearAutoencoder(nn.Module):
    """Linear AE behaves similarly to PCA while remaining an AE model."""

    def __init__(self, input_size, bottleneck):
        super().__init__()
        self.encoder = nn.Linear(input_size, bottleneck)
        self.decoder = nn.Linear(bottleneck, input_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: Train an autoencoder on concatenated frame vectors and "
            "save bottleneck representations for Step 5."
        )
    )
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of the training set used as validation data.",
    )
    return parser.parse_args()


def encode_in_batches(model, vectors, batch_size, device):
    dataset = TensorDataset(torch.from_numpy(vectors.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    encoded = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z = model.encode(batch_x)
            encoded.append(z.cpu().numpy())
    return np.concatenate(encoded, axis=0)


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print("\nLoading AE vectors...")
    train_vectors = np.load(f"{RESULTS_DIR}/train_ae_vectors.npy").astype(np.float32)
    test_vectors = np.load(f"{RESULTS_DIR}/test_ae_vectors.npy").astype(np.float32)
    train_labels = np.load(f"{RESULTS_DIR}/train_labels.npy")
    test_labels = np.load(f"{RESULTS_DIR}/test_labels.npy")

    print(f"Train shape : {train_vectors.shape}")
    print(f"Test shape  : {test_vectors.shape}")
    print(f"Train labels: {train_labels.shape}")
    print(f"Test labels : {test_labels.shape}")

    input_size = int(train_vectors.shape[1])
    if test_vectors.shape[1] != input_size:
        raise ValueError(
            "Input vector size mismatch between train and test: "
            f"train={input_size}, test={test_vectors.shape[1]}"
        )
    if args.bottleneck <= 0 or args.bottleneck > input_size:
        raise ValueError(f"bottleneck must be in [1, {input_size}]")
    if not (0.0 < args.val_ratio < 0.5):
        raise ValueError("val-ratio must be in the range (0, 0.5)")

    print(f"\nInput size      : {input_size}")
    print(f"Bottleneck size : {args.bottleneck}")
    print(f"Compression     : {input_size / args.bottleneck:.2f}x")

    print("\nCreating validation split from training data...")
    train_fit_vectors, val_vectors, train_fit_labels, val_labels = train_test_split(
        train_vectors,
        train_labels,
        test_size=args.val_ratio,
        stratify=train_labels,
        random_state=42,
    )
    print(f"Fit split shape: {train_fit_vectors.shape}")
    print(f"Val split shape: {val_vectors.shape}")
    print(f"Fit labels    : {train_fit_labels.shape}")
    print(f"Val labels    : {val_labels.shape}")

    print("\nNormalizing with fit-split feature statistics...")
    train_feature_mean = train_fit_vectors.mean(axis=0, keepdims=True)
    train_feature_std = train_fit_vectors.std(axis=0, keepdims=True) + 1e-8

    train_fit_vectors = (train_fit_vectors - train_feature_mean) / train_feature_std
    val_vectors = (val_vectors - train_feature_mean) / train_feature_std
    train_vectors = (train_vectors - train_feature_mean) / train_feature_std
    test_vectors = (test_vectors - train_feature_mean) / train_feature_std

    train_fit_vectors = train_fit_vectors.astype(np.float32)
    val_vectors = val_vectors.astype(np.float32)
    train_vectors = train_vectors.astype(np.float32)
    test_vectors = test_vectors.astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(train_fit_vectors))
    val_dataset = TensorDataset(torch.from_numpy(val_vectors))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    model = LinearAutoencoder(input_size=input_size, bottleneck=args.bottleneck).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    print("\n" + "=" * 50)
    print("TRAINING AUTOENCODER")
    print("=" * 50)

    train_losses = []
    val_losses = []
    best_loss = float("inf")
    training_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_x)
                epoch_val_loss += loss.item()

        avg_val = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), f"{RESULTS_DIR}/autoencoder_best.pth")

        if (epoch + 1) % 20 == 0:
            elapsed = (time.time() - training_start) / 60.0
            print(
                f"Epoch [{epoch + 1:3d}/{args.epochs}]  "
                f"Train: {avg_train:.6f}  "
                f"Val: {avg_val:.6f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}  "
                f"Elapsed: {elapsed:.1f} min"
            )

    training_time = (time.time() - training_start) * 1000.0
    print(f"\nTotal training time: {training_time:.1f} ms")
    print(f"Best val loss      : {best_loss:.6f}")

    print("\nExtracting bottleneck vectors from best model...")
    model.load_state_dict(torch.load(f"{RESULTS_DIR}/autoencoder_best.pth", map_location=device))
    train_bottleneck = encode_in_batches(model, train_vectors, args.batch_size, device)
    test_bottleneck = encode_in_batches(model, test_vectors, args.batch_size, device)

    print(f"Train bottleneck shape: {train_bottleneck.shape}")
    print(f"Test bottleneck shape : {test_bottleneck.shape}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Train/Val Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ae_loss_curve.png")
    plt.close()

    np.save(f"{RESULTS_DIR}/train_bottleneck.npy", train_bottleneck)
    np.save(f"{RESULTS_DIR}/test_bottleneck.npy", test_bottleneck)
    np.save(f"{RESULTS_DIR}/train_feature_mean.npy", train_feature_mean.astype(np.float32))
    np.save(f"{RESULTS_DIR}/train_feature_std.npy", train_feature_std.astype(np.float32))

    summary = {
        "method": "autoencoder_linear_strict",
        "input_size": int(input_size),
        "bottleneck": int(args.bottleneck),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "val_ratio": float(args.val_ratio),
        "fit_samples": int(len(train_fit_vectors)),
        "val_samples": int(len(val_vectors)),
        "best_val_loss": float(best_loss),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "training_time_ms": float(training_time),
    }
    np.save(f"{RESULTS_DIR}/ae_summary.npy", summary)

    print("\nSaved files:")
    print(f"  {RESULTS_DIR}/autoencoder_best.pth")
    print(f"  {RESULTS_DIR}/train_bottleneck.npy")
    print(f"  {RESULTS_DIR}/test_bottleneck.npy")
    print(f"  {RESULTS_DIR}/train_feature_mean.npy")
    print(f"  {RESULTS_DIR}/train_feature_std.npy")
    print(f"  {RESULTS_DIR}/ae_loss_curve.png")
    print(f"  {RESULTS_DIR}/ae_summary.npy")
    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()