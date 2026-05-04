import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Learns WHERE to focus in the feature map.
    Applies average and max pooling along channel axis,
    concatenates them, then uses a conv layer to produce
    a spatial attention map.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            in_channels  = 2,   # avg + max pooled
            out_channels = 1,
            kernel_size  = kernel_size,
            padding      = kernel_size // 2,
            bias         = False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling across channels → [B, 1, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Max pooling across channels → [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate → [B, 2, H, W]
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Generate attention map → [B, 1, H, W]
        attention_map = self.sigmoid(self.conv(pooled))

        # Apply attention to input
        return x * attention_map


class LeNet5WithAttention(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5WithAttention, self).__init__()

        # CONV Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Spatial Attention after CONV1
        self.attention1 = SpatialAttention(kernel_size=7)

        # CONV Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Spatial Attention after CONV2
        self.attention2 = SpatialAttention(kernel_size=5)

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        # Block 1 + Attention
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.attention1(x)

        # Block 2 + Attention
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.attention2(x)

        # Classifier
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    print("LeNet-5 WITH Attention:")
    model_att = LeNet5WithAttention()
    total_params_att = sum(p.numel() for p in model_att.parameters())
    print(f"Total parameters: {total_params_att:,}")
    print("✅ Step 3 Complete — LeNet-5 + Attention built!")