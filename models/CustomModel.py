import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A 1D implementation of the Residual Block, designed for building deep networks that process sequence data.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # [Core Change] Using Conv1d and BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        # If the input and output channels do not match, or the stride is not 1,
        # a 1x1 1D convolution is needed to match the dimensions.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # The forward propagation logic is identical to the 2D version.
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomModel(nn.Module):
    """
    A custom 1D ResNet model, specifically designed for processing 1D signal/sequence data.
    - It accepts input of shape (batch_size, num_channels, signal_length), e.g., (B, 1, 4096).
    """

    def __init__(self, num_classes=3, in_channels=1):
        super(CustomModel, self).__init__()

        # [Core Change] Using Conv1d and BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        # Using the 1D version of the residual block
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)

        # [Core Change] Using a 1D adaptive average pooling layer, with an output length of 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # The fully connected layer remains unchanged
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Ensure the input is 3D (B, C, L)
        # print(f"Input shape to model: {x.shape}") # (Optional debug statement)

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)

        # Shape after pooling: (B, 128, L') -> (B, 128, 1)
        out = self.avg_pool(out)

        # Flatten for the fully connected layer: (B, 128, 1) -> (B, 128)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out


# Explicitly declare the public interface of this module
__all__ = ['CustomModel']
