import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

_last_layer = False
num_layer_test = 0

class OnlyInvQ_AcosW_Act_Merge(nn.Module):
    def __init__(self, M, B, Qmax):
        super(OnlyInvQ_AcosW_Act_Merge, self).__init__()
        self.M = float(M)
        self.B = torch.tensor(B, dtype=torch.float32) if not torch.is_tensor(B) else B
        self.Qmax = float(Qmax)
        self.register_buffer('B_buffer', self.B)

    def forward(self, X):
        """
            Exact implementation based on the MATLAB code logic:
            Z = (X.*layer.M + layer.B);
            Z(Z > layer.Qmax) = layer.Qmax;
            Z(Z < -layer.Qmax) = -layer.Qmax;
        """
        # First, perform scaling and bias operations
        Z = X * self.M + self.B_buffer.view(1, -1, 1)  # Ensure correct broadcasting dimensions
        # Then perform clamping
        max = torch.max(Z)
        min = torch.min(Z)
        Z = torch.where(Z > self.Qmax, torch.tensor(self.Qmax, device=Z.device, dtype=Z.dtype), Z)
        Z = torch.where(Z < -self.Qmax, torch.tensor(-self.Qmax, device=Z.device, dtype=Z.dtype), Z)
        return Z

class InvQ_AcosW_Act_MergeQ(nn.Module):
    def __init__(self, M, B, nBit_shift, Qmax):
        super(InvQ_AcosW_Act_MergeQ, self).__init__()
        self.M = float(M)  # Ensure M is a scalar float
        self.B = torch.tensor(B, dtype=torch.float32) if not torch.is_tensor(B) else B
        self.nBit_shift = nBit_shift
        self.Qmax = Qmax
        self.register_buffer('B_buffer', self.B)

    def forward(self, X):
        # Correct broadcasting method
        intZ = (torch.mul(X, self.M) + self.B_buffer.view(1, -1, 1)).to(torch.int64)
        Z = (intZ >> self.nBit_shift).to(torch.float32) + ((intZ >> (self.nBit_shift - 1)) & 1).to(torch.float32)
        max = torch.max(Z)
        min = torch.min(Z)
        Qmax = self.Qmax
        Z = torch.clamp(Z, -self.Qmax, self.Qmax)
        return Z


class CosConvLayerBaselineSetW(nn.Module):
    def __init__(self, filterLength, numFilters, numChannels, numStride, params):
        super(CosConvLayerBaselineSetW, self).__init__()

        self.numFilters = numFilters
        self.numChannels = numChannels
        self.numStride = numStride
        self.filterLength = filterLength
        self.Qmax_w = params['Qmax_w']

        # Register parameters
        self.register_buffer('w', params['w'])  # shape: (1, 1, numFilters, numChannels)
        self.register_buffer('QX_LUT', params['QX_LUT'])  # shape: (4096, filterLength)
        self.register_buffer('A', params['A'])  # shape: (1, 1, numFilters, numChannels)

        # Precompute padding during initialization
        self.padding = self.filterLength // 2

        # Precompute the weight matrix
        self._precompute_weights()

    def _precompute_weights(self):
        """Precompute convolution weights using vectorized operations"""
        # Adjust dimensions and ensure tensor continuity
        w = self.w.squeeze(0).squeeze(0).permute(1, 0).contiguous()  # (numChannels, numFilters)
        A = self.A.squeeze(0).squeeze(0).permute(1, 0).contiguous()  # (numChannels, numFilters)

        # Compute indices and ensure continuity
        indices = (w + self.Qmax_w + 1).long().contiguous()  # (numChannels, numFilters)

        # Use reshape instead of view for safety
        flat_indices = indices.reshape(-1)  # (numChannels * numFilters)
        lut_values = self.QX_LUT[flat_indices]  # (numChannels * numFilters, filterLength)

        # Reshape LUT values, using reshape for safe operation
        lut_values = lut_values.reshape(self.numChannels, self.numFilters, self.filterLength)
        A = A.unsqueeze(-1)  # (numChannels, numFilters, 1)

        # Compute all weights at once
        W = (A * lut_values).permute(1, 0, 2).contiguous()  # (numFilters, numChannels, filterLength)

        # Save the precomputed weights
        self.register_buffer('precomputed_weights', W)

    def forward(self, X):
        """Optimized forward propagation"""
        # Directly use precomputed weights for convolution
        return F.conv1d(X,
                        self.precomputed_weights,
                        bias=None,
                        stride=self.numStride,
                        padding=self.padding,
                        dilation=1,
                        groups=1)


class BasicConvLayer(nn.Module):
    def __init__(self, filterLength, numFilters, numChannels, numStride, params, M, B, nBit_shift, Qmax,
                 is_last_layer=False):

        super(BasicConvLayer, self).__init__()

        self.cos_conv_layer = CosConvLayerBaselineSetW(filterLength, numFilters, numChannels, numStride, params)

        self.invq_acosw_act_mergeq = InvQ_AcosW_Act_MergeQ(M, B, nBit_shift, Qmax)

        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, X):
        global num_layer_test
        X = self.cos_conv_layer(X)
        X = self.invq_acosw_act_mergeq(X)
        X = self.pooling(X)

        # print(f"Layer{num_layer_test}: [{X.min():.4f}, {X.max():.4f}]")
        return X


class QuantizedCosCNN(nn.Module):
    def __init__(self, input_length, num_classes, filter_length, num_filters_list,
                 params_list, M_list, B_list, nBit_shift_list, Qmax_list, num_channels,
                 initial_M, initial_B, initial_nBit_shift, initial_Qmax, final_M, final_B):
        super(QuantizedCosCNN, self).__init__()

        # Keep the original initial layer
        self.initial_invq_acosw_act_mergeq = InvQ_AcosW_Act_MergeQ(
            M=initial_M,
            B=initial_B,
            nBit_shift=initial_nBit_shift,
            Qmax=initial_Qmax
        )

        # Keep the original feature extractor construction code
        layers = []
        in_channels = num_channels[0]

        for i in range(len(num_filters_list)):
            conv_name = f'conv{i}'
            layers.append(
                (conv_name, BasicConvLayer(
                    filterLength=filter_length,
                    numFilters=num_filters_list[i],
                    numChannels=in_channels,
                    numStride=1,
                    params=params_list[i],
                    M=M_list[i],
                    B=B_list[i],
                    nBit_shift=nBit_shift_list[i],
                    Qmax=Qmax_list[i],
                    is_last_layer=(i == len(num_filters_list) - 1)
                ))
            )
            in_channels = num_filters_list[i]

        self.feature_extractor = nn.Sequential(OrderedDict(layers))

        # Add new OnlyInvQ_AcosW_Act_Merge layer
        self.final_quant = OnlyInvQ_AcosW_Act_Merge(
            M=final_M,
            B=final_B,
            Qmax=initial_Qmax
        )

        # Dynamically compute the input dimension for the fully connected layer
        with torch.no_grad():
            x = torch.zeros(1, num_channels[0], input_length)
            x = self.initial_invq_acosw_act_mergeq(x)
            x = self.feature_extractor(x)
            x = self.final_quant(x)  # Add processing for the new layer
            # print(f"After feature_extractor: {x.shape}")
            fc_input_dim = x.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, num_classes)
        )

    def forward(self, x):
        x = self.initial_invq_acosw_act_mergeq(x)
        x = self.feature_extractor(x)
        x = self.final_quant(x)  # Add processing for the new layer
        x = self.classifier(x)
        return x
