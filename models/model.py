import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class CosConvLayer(nn.Module):
    """Implementation of Cosine Convolution Layer

    This layer uses the cosine function as the fundamental operation unit, incorporating learnable
    amplitude parameter A and frequency parameter w. The position encoding x is utilized to generate
    the final convolution kernel.

    Attributes:
        name (str): Name of the layer
        num_channels (int): Number of input channels
        num_filters (int): Number of convolution filters
        filter_length (int): Length of the convolution kernel
        A (nn.Parameter): Amplitude parameter
        w (nn.Parameter): Frequency parameter
        x_buffer (tensor): Position encoding
    """

    def __init__(self, name, filter_length, num_channels, num_filters, num_stride):
        super(CosConvLayer, self).__init__()
        self.name = name
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.num_stride = num_stride
        self.filter_length = filter_length

        # Initialize learnable parameters A and w
        self.A = nn.Parameter(torch.randn(num_filters, num_channels))
        self.w = nn.Parameter(torch.randn(num_filters, num_channels))

        # Position encoding x
        x = torch.arange(0, filter_length).float() - (filter_length - 1) / 2
        x = x.view(1, 1, filter_length)
        self.register_buffer('x_buffer', x)

        self.Bias = None

    def forward(self, X):
        """Forward propagation function

        Args:
            X (tensor): Input data with shape [batch_size, num_channels, input_length]

        Returns:
            tensor: Convolution output with shape [batch_size, num_filters, output_length]
        """
        device = X.device
        x = self.x_buffer.to(device)

        # Compute convolution kernel
        A = self.A.unsqueeze(-1)
        w = self.w.abs().unsqueeze(-1)
        W = A * torch.cos(w * x)

        # Perform convolution operation
        Z = F.conv1d(X, W, stride=self.num_stride, padding=self.filter_length // 2)
        if self.Bias is not None:
            Z = Z + self.Bias.view(1, -1, 1).to(device)
        return Z


class CosCNN(nn.Module):
    """
        Cosine Convolution-based CNN Network

        The network consists of multiple cosine convolution layers, batch normalization layers,
        and max-pooling layers, followed by a fully connected layer for classification.

        Attributes:
            feature_extractor (nn.Sequential): Feature extractor consisting of multiple convolution,
                batch normalization, and pooling layers.
            classifier (nn.Sequential): Classifier consisting of fully connected layers.
    """

    def __init__(self, input_length, in_channels, num_classes, filter_length, num_filters_list):
        super(CosCNN, self).__init__()
        layers = []
        # in_channels = 1

        # Construct feature extractor
        for i, num_filters in enumerate(num_filters_list):
            conv_name = f'conv{i}'
            bn_name = f'bn{i}'
            pool_name = f'pool{i}'

            # Add cosine convolution layer
            layers.append((conv_name, CosConvLayer(conv_name, filter_length=filter_length,
                                                   num_channels=in_channels,
                                                   num_filters=num_filters, num_stride=1)))
            # Add batch normalization layer
            layers.append((bn_name, nn.BatchNorm1d(num_filters)))
            # Add max pooling layer
            layers.append((pool_name, nn.MaxPool1d(kernel_size=2)))

            in_channels = num_filters

        self.feature_extractor = nn.Sequential(OrderedDict(layers))

        # Dynamically compute classifier input dimension
        with torch.no_grad():
            x = torch.zeros(1, 1, input_length)
            x = self.feature_extractor(x)
            fc_input_dim = x.shape[1] * x.shape[2]

        # Construct classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, num_classes)
        )

    def forward(self, x):
        """
            Forward propagation function

            Args:
                x (tensor): Input data with shape [batch_size, 1, input_length]

            Returns:
                tensor: Classification output with shape [batch_size, num_classes]
        """
        # print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


# Export classes
__all__ = ['CosConvLayer', 'CosCNN']