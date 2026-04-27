"""
model.py
--------
Defines the CNN architecture for Arabic digit speech recognition.

The network treats each log-mel spectrogram (1 × 128 × 128) as a grayscale
image and learns spatial patterns (formants, harmonics, temporal transitions)
directly from the pixel values — exactly like an image classifier.

Architecture  –  DigitCNN
─────────────────────────
Input  :  (B, 1, 128, 128)

Block 1 │ Conv2d(1  → 32, 3×3) → BN → ReLU → Conv2d(32 → 32, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
Block 2 │ Conv2d(32 → 64, 3×3) → BN → ReLU → Conv2d(64 → 64, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
Block 3 │ Conv2d(64 →128, 3×3) → BN → ReLU → Conv2d(128→128, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)

Flatten → FC(128*14*14 → 512) → BN → ReLU → Dropout(0.5)
        → FC(512 → 10)  ← logits (no softmax; use CrossEntropyLoss)

Output :  (B, 10)  — raw logits

Why this design?
  • Double-conv blocks (VGG-style) learn richer features before downsampling.
  • BatchNorm after every conv stabilises training on a small dataset (1200 samples).
  • Progressive filter growth (32 → 64 → 128) captures low-level edges first,
    then higher-level spectro-temporal patterns.
  • Dropout at 0.25 (conv) and 0.5 (FC) reduces overfitting.
  • No softmax in forward() — PyTorch's CrossEntropyLoss expects raw logits.
"""

import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    """
    CNN for classifying log-mel spectrograms into 10 Arabic digit classes.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 10 for digits 0–9).
    dropout_conv : float
        Dropout probability after each convolutional block (default 0.25).
    dropout_fc : float
        Dropout probability after the first fully-connected layer (default 0.5).
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_conv: float = 0.25,
        dropout_fc:   float = 0.50,
    ):
        super().__init__()

        # ── Convolutional blocks ──────────────────────────────────────────
        self.block1 = self._conv_block(in_ch=1,   out_ch=32,  dropout=dropout_conv)
        self.block2 = self._conv_block(in_ch=32,  out_ch=64,  dropout=dropout_conv)
        self.block3 = self._conv_block(in_ch=64,  out_ch=128, dropout=dropout_conv)

        # After 3 × MaxPool(2×2): 128 → 64 → 32 → 16
        # But each double-conv with padding=1 keeps spatial size,
        # so after 3 pools: 128 / 8 = 16
        # Feature map: 128 channels × 16 × 16 = 32768
        self._flat_features = 128 * 16 * 16

        # ── Fully-connected head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),   # raw logits
        )

    # ── builder for a double-conv block ──────────────────────────────────
    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
        """
        Two consecutive Conv → BN → ReLU layers followed by MaxPool and Dropout.

        Spatial size is preserved through each conv (padding=1) and halved by
        the MaxPool at the end.
        """
        return nn.Sequential(
            # first conv
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # second conv
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # downsample
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (B, 1, 128, 128)

        Returns
        -------
        torch.Tensor  shape (B, 10)  — raw logits
        """
        x = self.block1(x)   # (B,  32, 64, 64)
        x = self.block2(x)   # (B,  64, 32, 32)
        x = self.block3(x)   # (B, 128, 16, 16)
        x = self.classifier(x)  # (B, 10)
        return x


# ─────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────
def get_model(device: torch.device, **kwargs) -> DigitCNN:
    """
    Instantiate DigitCNN, move it to `device`, and print a parameter summary.

    Any keyword arguments are forwarded to DigitCNN.__init__
    (e.g. dropout_conv=0.3, dropout_fc=0.4).
    """
    model = DigitCNN(**kwargs).to(device)

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DigitCNN  |  total params: {total:,}  |  trainable: {trainable:,}")
    return model


# ─────────────────────────────────────────────
#  Quick sanity check  (run this file directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] Using device: {device}")

    model = get_model(device)

    # Dummy forward pass
    dummy = torch.randn(4, 1, 128, 128).to(device)   # batch of 4 spectrograms
    out   = model(dummy)
    print(f"[model] Input  shape : {dummy.shape}")
    print(f"[model] Output shape : {out.shape}")    # expected: torch.Size([4, 10])
    print(model)