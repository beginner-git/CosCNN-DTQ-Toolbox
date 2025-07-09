import os
import sys

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.model import CosCNN, CosConvLayer


class FeatureParameterExtractor(nn.Module):
    """Feature and Parameter Extractor Class

    This class is responsible for:
    1. Extracting parameters from each layer of the model (both original and merged parameters).
    2. Extracting feature maps at the correct locations (after BN layers).
    3. Handling the dimension transformation and merge calculations for parameters.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = OrderedDict()
        self.parameters_dict = OrderedDict()
        self.current_conv_index = 0

        # Extract parameters and register feature extraction hooks
        self._extract_parameters()
        self._register_hooks()

    def _register_hooks(self):
        """Registers hook functions for feature extraction.

        Registers a hook after each BN layer to get the complete feature map.
        """
        self.current_conv_index = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                module.register_forward_hook(
                    self.get_feature_hook(f'conv{self.current_conv_index}'))
                self.current_conv_index += 1

    def _extract_parameters(self):
        """Extracts and processes model parameters.

        This includes:
        1. Extracting original convolutional and BN layer parameters.
        2. Calculating merged parameters (considering the effect of the BN layer).
        3. Extracting FC layer parameters.
        """
        self.current_conv_index = 0
        prev_conv = None

        for name, module in self.model.named_modules():
            if isinstance(module, CosConvLayer):
                # Extract convolutional layer parameters
                prev_conv = module
                # Store original parameters
                self.parameters_dict[f'conv{self.current_conv_index}_A'] = module.A.detach().cpu().numpy()
                self.parameters_dict[f'conv{self.current_conv_index}_w'] = torch.abs(module.w).detach().cpu().numpy()
                self.parameters_dict[f'conv{self.current_conv_index}_x'] = module.x_buffer.cpu().numpy()

            elif isinstance(module, nn.BatchNorm1d) and prev_conv is not None:
                # Extract BN layer parameters
                self.parameters_dict[f'bn{self.current_conv_index}_weight'] = module.weight.detach().cpu().numpy()
                self.parameters_dict[f'bn{self.current_conv_index}_bias'] = module.bias.detach().cpu().numpy()
                self.parameters_dict[
                    f'bn{self.current_conv_index}_running_mean'] = module.running_mean.detach().cpu().numpy()
                self.parameters_dict[
                    f'bn{self.current_conv_index}_running_var'] = module.running_var.detach().cpu().numpy()

                # Calculate merged parameters
                epsilon = 1e-5
                scale = module.weight.detach().cpu().numpy()
                offset = module.bias.detach().cpu().numpy()
                running_mean = module.running_mean.detach().cpu().numpy()
                running_var = module.running_var.detach().cpu().numpy()

                # Calculate BN scaling factor
                a = scale / np.sqrt(running_var + epsilon)
                b = offset - running_mean * a

                # Merged parameter calculation
                A = prev_conv.A.detach().cpu().numpy()
                merged_A = A * a[:, np.newaxis]  # Broadcast multiplication, handle dimensions correctly

                # Store merged parameters
                self.parameters_dict[f'conv{self.current_conv_index}_merged_A'] = merged_A
                self.parameters_dict[f'conv{self.current_conv_index}_merged_bias'] = b

                self.current_conv_index += 1
                prev_conv = None

            elif isinstance(module, nn.Linear):
                # Extract weights and biases of the FC layer
                self.parameters_dict[f'fc_weight'] = module.weight.detach().cpu().numpy()
                self.parameters_dict[f'fc_bias'] = module.bias.detach().cpu().numpy()

                self.current_conv_index += 1

    def get_feature_hook(self, layer_name):
        """Creates a hook function for feature extraction."""

        def hook(module, input, output):
            self.features[layer_name] = output.detach().cpu().numpy()

        return hook

    def forward(self, x):
        """Forward propagation function."""
        self.features.clear()
        output = self.model(x)
        return output
