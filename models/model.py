# CosCNN-DTQ Toolbox
# Copyright (C) 2024-2025 Guoyang Liu
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

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

    def __init__(self, name, filter_length, num_channels, num_filters, num_stride, groups: int = 1):
        super(CosConvLayer, self).__init__()
        self.name = name
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.num_stride = num_stride
        self.filter_length = filter_length
        self.groups = int(groups)
        if self.groups < 1:
            raise ValueError(f"groups must be >= 1, got {self.groups}")

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
        Z = F.conv1d(X, W, stride=self.num_stride, padding=self.filter_length // 2, groups=self.groups)
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

        initial_in_channels = in_channels

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
            x = torch.zeros(1, initial_in_channels, input_length)
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
                x (tensor): Input data with shape [batch_size, in_channels, input_length]

            Returns:
                tensor: Classification output with shape [batch_size, num_classes]
        """
        # print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x



class GroupCosCNN(nn.Module):
    """
    Group-wise Cosine Convolution-based CNN Network (GroupCosCNN)

    This network supports grouped cosine convolutions per layer.

    - `num_filters_list` is interpreted as **per-group** output channel counts for each layer.
      Therefore, total output channels of layer i is:
          out_channels_total = groups_i * num_filters_list[i]

    - `groups_list` allows configuring `groups` for each layer (same length as num_filters_list).
        * If groups_list is None -> all layers default to groups=in_channels (original behavior).
        * If groups_list[i] is None / "auto" -> inherit previous layer's groups
          (layer 0 inherits from in_channels).

    Expected input shape:
        x: [batch_size, in_channels, input_length]
    """

    def __init__(self, input_length, in_channels, num_classes, filter_length, num_filters_list, groups_list=None):
        super(GroupCosCNN, self).__init__()

        if in_channels is None:
            raise ValueError("in_channels must not be None")
        if int(in_channels) < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")

        self.in_channels = int(in_channels)
        self.per_group_filters = [int(x) for x in list(num_filters_list)]

        # Prepare groups_list aligned with layers
        if groups_list is None:
            groups_list = [None for _ in self.per_group_filters]
        else:
            groups_list = list(groups_list)
        if len(groups_list) < len(self.per_group_filters):
            groups_list += [None] * (len(self.per_group_filters) - len(groups_list))
        self.groups_list = groups_list[:len(self.per_group_filters)]

        layers = []

        in_channels_total = self.in_channels
        prev_groups = self.in_channels

        for i, num_filters_per_group in enumerate(self.per_group_filters):
            conv_name = f'conv{i}'
            bn_name = f'bn{i}'
            pool_name = f'pool{i}'

            g = self.groups_list[i]
            if g is None:
                g = prev_groups
            g = int(g)

            if g < 1:
                raise ValueError(f"groups must be >= 1 at layer {i}, got {g}")
            if in_channels_total % g != 0:
                raise ValueError(
                    f"Input channels ({in_channels_total}) is not divisible by groups ({g}) at layer {i}"
                )

            in_channels_per_group = in_channels_total // g
            total_out_channels = g * int(num_filters_per_group)

            layers.append((
                conv_name,
                CosConvLayer(
                    conv_name,
                    filter_length=filter_length,
                    num_channels=in_channels_per_group,          # channels per group
                    num_filters=total_out_channels,              # total output channels across all groups
                    num_stride=1,
                    groups=g
                )
            ))
            layers.append((bn_name, nn.BatchNorm1d(total_out_channels)))
            layers.append((pool_name, nn.MaxPool1d(kernel_size=2)))

            # Next layer total input channels equals this layer total output channels
            in_channels_total = int(total_out_channels)
            prev_groups = g

        self.feature_extractor = nn.Sequential(OrderedDict(layers))

        # Dynamically compute classifier input dimension
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, input_length)
            x = self.feature_extractor(x)
            fc_input_dim = x.shape[1] * x.shape[2]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


# Export classes
__all__ = ['CosConvLayer', 'CosCNN', 'GroupCosCNN']