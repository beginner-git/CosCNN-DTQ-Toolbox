import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class OnlyInvQ_AcosW_Act_Merge(nn.Module):
    def __init__(self, M, B, Qmax):
        super(OnlyInvQ_AcosW_Act_Merge, self).__init__()
        self.M = float(M)
        self.B = torch.tensor(B, dtype=torch.float32) if not torch.is_tensor(B) else B
        self.Qmax = float(Qmax)
        self.register_buffer('B_buffer', self.B)

    def forward(self, X):
        Z = X * self.M + self.B_buffer.view(1, -1, 1)
        Z = torch.where(Z > self.Qmax, torch.tensor(self.Qmax, device=Z.device, dtype=Z.dtype), Z)
        Z = torch.where(Z < -self.Qmax, torch.tensor(-self.Qmax, device=Z.device, dtype=Z.dtype), Z)
        return Z


class InvQ_AcosW_Act_MergeQ(nn.Module):
    def __init__(self, M, B, nBit_shift, Qmax):
        super(InvQ_AcosW_Act_MergeQ, self).__init__()
        self.M = float(M)
        self.B = torch.tensor(B, dtype=torch.float32) if not torch.is_tensor(B) else B
        self.nBit_shift = nBit_shift
        self.Qmax = Qmax
        self.register_buffer('B_buffer', self.B)

    def forward(self, X):
        intZ = (torch.mul(X, self.M) + self.B_buffer.view(1, -1, 1)).to(torch.int64)
        Z = (intZ >> self.nBit_shift).to(torch.float32) + ((intZ >> (self.nBit_shift - 1)) & 1).to(torch.float32)
        Z = torch.clamp(Z, -self.Qmax, self.Qmax)
        return Z


class CosConvLayerBaselineSetW(nn.Module):
    def __init__(self, filterLength, numFilters, numChannels, numStride, params, groups: int = 1):
        super(CosConvLayerBaselineSetW, self).__init__()

        self.numFilters = numFilters
        self.numChannels = numChannels
        self.numStride = numStride
        self.filterLength = filterLength
        self.groups = int(groups)
        if self.groups < 1:
            raise ValueError(f"groups must be >= 1, got {self.groups}")
        self.Qmax_w = params['Qmax_w']

        self.register_buffer('w', params['w'])
        self.register_buffer('QX_LUT', params['QX_LUT'])
        self.register_buffer('A', params['A'])

        self.padding = self.filterLength // 2

        self._precompute_weights()

    def _precompute_weights(self):
        """Precompute convolution weights using vectorized operations"""
        w = self.w.squeeze(0).squeeze(0).permute(1, 0).contiguous()
        A = self.A.squeeze(0).squeeze(0).permute(1, 0).contiguous()

        # Map w_q from [1, 2*Qmax_w+1] to LUT index [0, 2*Qmax_w]
        indices = (w - 1).long().contiguous()

        # Safety clamp to avoid out-of-bounds access
        indices = torch.clamp(indices, 0, self.QX_LUT.shape[0] - 1)

        flat_indices = indices.reshape(-1)
        lut_values = self.QX_LUT[flat_indices]

        lut_values = lut_values.reshape(self.numChannels, self.numFilters, self.filterLength)
        A = A.unsqueeze(-1)

        W = (A * lut_values).permute(1, 0, 2).contiguous()

        self.register_buffer('precomputed_weights', W)

    def forward(self, X):
        """Optimized forward propagation"""
        return F.conv1d(X,
                        self.precomputed_weights,
                        bias=None,
                        stride=self.numStride,
                        padding=self.padding,
                        dilation=1,
                        groups=self.groups)


class BasicConvLayer(nn.Module):
    def __init__(self, filterLength, numFilters, numChannels, numStride, params, M, B, nBit_shift, Qmax,
                 is_last_layer=False, groups: int = 1):

        super(BasicConvLayer, self).__init__()

        self.cos_conv_layer = CosConvLayerBaselineSetW(filterLength, numFilters, numChannels, numStride, params, groups=groups)
        self.invq_acosw_act_mergeq = InvQ_AcosW_Act_MergeQ(M, B, nBit_shift, Qmax)
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, X):
        X = self.cos_conv_layer(X)
        X = self.invq_acosw_act_mergeq(X)
        X = self.pooling(X)
        return X


class QuantizedCosCNN(nn.Module):
    def __init__(self, input_length, num_classes, filter_length, num_filters_list,
                 params_list, M_list, B_list, nBit_shift_list, Qmax_list, num_channels,
                 initial_M, initial_B, initial_nBit_shift, initial_Qmax, final_M, final_B, groups: int = 1, groups_list=None):
        super(QuantizedCosCNN, self).__init__()
        self.groups = int(groups)
        if self.groups < 1:
            raise ValueError(f"groups must be >= 1, got {self.groups}")

        # groups_list: per-layer groups; None means inherit previous layer's groups.
        if groups_list is None:
            self.groups_list = [None for _ in range(len(num_filters_list))]
        else:
            gl = list(groups_list)
            if len(gl) < len(num_filters_list):
                gl += [None] * (len(num_filters_list) - len(gl))
            self.groups_list = gl[:len(num_filters_list)]


        self.initial_invq_acosw_act_mergeq = InvQ_AcosW_Act_MergeQ(
            M=initial_M,
            B=initial_B,
            nBit_shift=initial_nBit_shift,
            Qmax=initial_Qmax
        )

        layers = []

        # For grouped conv:
        # - `num_channels[0]` is the *total* input channels.
        # - Each conv layer uses `groups` groups, so the *per-group* input channels is
        #   total_in_channels // groups.
        in_channels_total = int(num_channels[0])

        prev_groups = self.groups
        for i in range(len(num_filters_list)):
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
            if int(num_filters_list[i]) % g != 0:
                raise ValueError(
                    f"Output channels ({int(num_filters_list[i])}) is not divisible by groups ({g}) at layer {i}"
                )
            in_channels_per_group = in_channels_total // g

            conv_name = f'conv{i}'
            layers.append(
                (conv_name, BasicConvLayer(
                    filterLength=filter_length,
                    numFilters=num_filters_list[i],
                    numChannels=in_channels_per_group,
                    numStride=1,
                    params=params_list[i],
                    M=M_list[i],
                    B=B_list[i],
                    nBit_shift=nBit_shift_list[i],
                    Qmax=Qmax_list[i],
                    is_last_layer=(i == len(num_filters_list) - 1),
                    groups=g
                ))
            )
            prev_groups = g
            in_channels_total = int(num_filters_list[i])

        self.feature_extractor = nn.Sequential(OrderedDict(layers))

        self.final_quant = OnlyInvQ_AcosW_Act_Merge(
            M=final_M,
            B=final_B,
            Qmax=initial_Qmax
        )

        with torch.no_grad():
            x = torch.zeros(1, num_channels[0], input_length)
            x = self.initial_invq_acosw_act_mergeq(x)
            x = self.feature_extractor(x)
            x = self.final_quant(x)
            fc_input_dim = x.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, num_classes)
        )

    def forward(self, x):
        x = self.initial_invq_acosw_act_mergeq(x)
        x = self.feature_extractor(x)
        x = self.final_quant(x)
        x = self.classifier(x)
        return x