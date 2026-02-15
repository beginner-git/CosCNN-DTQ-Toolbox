import torch
import torch.nn as nn
import torch.nn.functional as F

# IMPORTANT:
# Use the toolbox's CosConvLayer so the CLI quantizer can detect it reliably.
from models.model import CosConvLayer


class ResidualBlock(nn.Module):
    """
    A 1D Residual Block using CosConvLayer (float in / float out).
    This keeps the same tensor length behavior as nn.Conv1d(kernel=3, padding=1)
    because CosConvLayer already applies same-padding internally: padding = filter_length // 2.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, name_prefix: str = "block"):
        super().__init__()

        # Main branch: two 3-tap CosConv
        self.conv1 = CosConvLayer(
            name=f"{name_prefix}_conv1",
            filter_length=3,
            num_channels=in_channels,
            num_filters=out_channels,
            num_stride=stride,
            groups=1,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = CosConvLayer(
            name=f"{name_prefix}_conv2",
            filter_length=3,
            num_channels=out_channels,
            num_filters=out_channels,
            num_stride=1,
            groups=1,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut branch: identity or 1x1 projection
        if stride != 1 or in_channels != out_channels:
            # Option A (recommended): keep projection as normal Conv1d (not quantized by CosConv-only quantizer)
            # This is the standard ResNet design and is usually more stable.
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(out_channels),
            )

            # Option B (fully CosConv): uncomment to replace projection with CosConvLayer(filter_length=1)
            # self.shortcut = nn.Sequential(
            #     CosConvLayer(
            #         name=f"{name_prefix}_shortcut",
            #         filter_length=1,
            #         num_channels=in_channels,
            #         num_filters=out_channels,
            #         num_stride=stride,
            #         groups=1,
            #     ),
            #     nn.BatchNorm1d(out_channels),
            # )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class CustomModel(nn.Module):
    """
    A small 1D ResNet-like model using CosConvLayer.

    Input:  (B, in_channels, signal_length) e.g., (B, 1, 4096)
    Output: (B, num_classes)

    Notes:
    - input_length is accepted for CLI compatibility, but is not required due to AdaptiveAvgPool1d(1).
    - This model is intended to demonstrate CosConv-only quantization (float-in/float-out per CosConv).
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 1, input_length: int = 4096, **kwargs):
        super().__init__()

        # Stem CosConv (kernel_size=3 equivalent)
        self.conv1 = CosConvLayer(
            name="stem_conv1",
            filter_length=3,
            num_channels=in_channels,
            num_filters=64,
            num_stride=1,
            groups=1,
        )
        self.bn1 = nn.BatchNorm1d(64)

        # Two residual blocks (same as your original structure)
        self.layer1 = ResidualBlock(64, 64, stride=1, name_prefix="layer1")
        self.layer2 = ResidualBlock(64, 128, stride=2, name_prefix="layer2")

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)          # (B, 128, 1)
        out = out.view(out.size(0), -1)   # (B, 128)
        out = self.fc(out)                # (B, num_classes)
        return out


__all__ = ["CustomModel", "ResidualBlock"]
