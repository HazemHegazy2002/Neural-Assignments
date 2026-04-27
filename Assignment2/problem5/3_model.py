"""
model.py
--------
LeNet-5 CNN architecture for ReducedMNIST digit recognition.

Original LeNet-5 (LeCun et al., 1998) was designed for 32x32 images.
We adapt it for 28x28 MNIST images by adjusting the first layer padding.

Architecture:
    Input  : (B, 1, 28, 28)

    CONV1  : Conv2d(1->6,  5x5, padding=2) -> (B, 6,  28, 28)
             BatchNorm -> ReLU
             AvgPool(2x2)                  -> (B, 6,  14, 14)

    CONV2  : Conv2d(6->16, 5x5, padding=0) -> (B, 16, 10, 10)
             BatchNorm -> ReLU
             AvgPool(2x2)                  -> (B, 16,  5,  5)

    FLATTEN: 16 * 5 * 5 = 400

    FC1    : Linear(400 -> 120) -> ReLU
    FC2    : Linear(120 ->  84) -> ReLU
    FC3    : Linear( 84 ->  10)  <- raw logits (no softmax)

    Output : (B, 10)

Why LeNet-5 for this problem?
    - Designed specifically for digit recognition
    - Lightweight (only ~60K parameters) -> trains fast on small datasets
    - Two conv layers capture local stroke features then combine them
    - Well proven on MNIST benchmarks
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 adapted for 28x28 grayscale images.

    Parameters
    ----------
    num_classes : int  (default 10 for digits 0-9)
    use_batchnorm : bool
        If True, adds BatchNorm after each conv layer.
        Helps stabilize training on small datasets like ReducedMNIST.
    """

    def __init__(self, num_classes=10, use_batchnorm=True):
        super().__init__()

        # ── CONV Block 1 ──────────────────────────────────────────────
        # padding=2 on a 28x28 image keeps output at 28x28 (same as 32x32 input)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),   # -> (B, 6, 14, 14)
        )

        # ── CONV Block 2 ──────────────────────────────────────────────
        # no padding: 14x14 -> 10x10 after 5x5 conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.BatchNorm2d(16) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),   # -> (B, 16, 5, 5)
        )

        # ── Fully Connected Head ──────────────────────────────────────
        # 16 * 5 * 5 = 400 flattened features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),   # raw logits
        )

    def forward(self, x):
        """
        x : (B, 1, 28, 28)
        returns : (B, 10) raw logits
        """
        x = self.conv1(x)       # (B,  6, 14, 14)
        x = self.conv2(x)       # (B, 16,  5,  5)
        x = self.classifier(x)  # (B, 10)
        return x


# -------------------------------------------------
#  Factory function
# -------------------------------------------------
def get_model(device):
    """
    Instantiate LeNet-5, move to device, print parameter count.
    """
    model = LeNet5().to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] LeNet-5 | total params: {total:,} | trainable: {trainable:,}")
    return model


# -------------------------------------------------
#  Run standalone - sanity check
# -------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] Device: {device}")

    model = get_model(device)

    # Verify shapes with a dummy forward pass
    dummy = torch.randn(8, 1, 28, 28).to(device)
    out   = model(dummy)

    print(f"[model] Input  shape : {dummy.shape}")
    print(f"[model] Output shape : {out.shape}")    # expected: (8, 10)
    print(model)