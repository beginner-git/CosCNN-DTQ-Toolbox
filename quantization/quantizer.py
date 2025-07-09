import numpy as np
from threshold_finder import FindThreshold
from utils_quantization import validate_quantized_model
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def getQ(n):
    Q = (2 ** n - 1) // 2
    return Q


class NetworkQuantizer:
    def __init__(self, parameters_dict, nBit_params, features_dict, num_layers, w_max, ifold):
        """
        Initializes the network quantizer.

        Args:
            parameters_dict (dict): Dictionary of original network parameters.
            nBit_params (dict): Dictionary of quantization bit-width parameters.
            features_dict (dict): Dictionary of features/activations.
            num_layers (int): Total number of layers.
            w_max (float): Maximum weight value.
        """
        self.parameters_dict = parameters_dict
        self.nBit_params = nBit_params
        self.features_dict = features_dict
        self.num_layers = num_layers
        self.w_max = w_max
        self.ifold = ifold

        # Initialize storage structures
        self.conv_para = {}
        self.Q_params = {}
        self.QAct_para = {}
        self.QX_LUT = []

        for iLayer in range(num_layers):
            self.conv_para[f'conv{iLayer}_A'] = parameters_dict[f'conv{iLayer}_A']
            self.conv_para[f'conv{iLayer}_w'] = np.abs(parameters_dict[f'conv{iLayer}_w'])
            self.conv_para[f'conv{iLayer}_x'] = parameters_dict[f'conv{iLayer}_x']




    def _merge_bn_layer(self, iLayer):
        """Merges Batch Normalization layer parameters."""
        [self.nBit_w, self.nBit_wx, self.nBit_A, self.nBit_Act, self.nBit_shift] = [
            int(self.nBit_params[f'conv{iLayer}_nBit_w']),
            int(self.nBit_params[f'conv{iLayer}_nBit_wx']),
            int(self.nBit_params[f'conv{iLayer}_nBit_A']),
            int(self.nBit_params[f'conv{iLayer}_nBit_Act']),
            int(self.nBit_params[f'conv{iLayer}_nBit_shift'])]

        [self.Qmax_w, self.Qmax_wx, self.Qmax_A, self.Qmax_Act, self.Qmax_shift] = [getQ(self.nBit_w),
                                                                                    getQ(self.nBit_wx),
                                                                                    getQ(self.nBit_A),
                                                                                    getQ(self.nBit_Act),
                                                                                    getQ(self.nBit_shift)]

        # Fusion of the BN layer
        EPSILON = 1e-5
        a = self.parameters_dict[f'bn{iLayer}_weight'][:, np.newaxis] / (
                (self.parameters_dict[f'bn{iLayer}_running_var'][:, np.newaxis] + EPSILON) ** 0.5)
        b = self.parameters_dict[f'bn{iLayer}_bias'][:, np.newaxis] - self.parameters_dict[f'bn{iLayer}_running_mean'][
                                                                      :,
                                                                      np.newaxis] * a

        self.conv_para[f'conv{iLayer}_A_merged'] = self.conv_para[f'conv{iLayer}_A'] * a
        self.conv_para[f'conv{iLayer}_merged_bias'] = b
        # Set convBias
        self.Q_params[f'conv{iLayer}_convBias'] = b  # Add this line

        num_merge_channel = np.size(self.parameters_dict[f'bn{iLayer}_weight'])
        for j in range(num_merge_channel):
            a = self.parameters_dict[f'bn{iLayer}_weight'][iLayer] / (
                    (self.parameters_dict[f'bn{iLayer}_running_var'][j] + EPSILON) ** 0.5
            )
            b = self.parameters_dict[f'bn{iLayer}_bias'][j] - self.parameters_dict[f'bn{iLayer}_running_mean'][j] * a

            self.conv_para[f'conv{iLayer}_A_merged'][j][:] = self.conv_para[f'conv{iLayer}_A'][j][:] * a
            self.conv_para[f'conv{iLayer}_merged_bias'][j] = b
            # Update convBias
            self.Q_params[f'conv{iLayer}_convBias'][j] = b  # Add this line

        Rmax = self.w_max  # w_max remains unchanged
        curConvS_w = (2 * self.Qmax_w + 1) / Rmax

        self.Q_params[f'conv{iLayer}_w_q'] = np.round(
            np.abs(self.conv_para[f'conv{iLayer}_w']) * curConvS_w)  # Use round instead of integer division
        self.Q_params[f'conv{iLayer}_w_q'] = self.Q_params[f'conv{iLayer}_w_q'] - (self.Qmax_w + 1)  # Centering


    def _generate_cosine_lut(self, iLayer):
        """Generates a cosine lookup table."""

        curConvS_w = (2 * self.Qmax_w + 1) / self.w_max
        # Cosine lookup table
        Rmax = 1
        curConvS_wx = self.Qmax_wx / Rmax

        # Save relevant parameters
        self.Q_params[f'conv{iLayer}_Qmax_wx'] = curConvS_wx
        self.Q_params[f'conv{iLayer}_Qscale_wx'] = curConvS_wx  # Add this line of code

        X = self.conv_para[f'conv{iLayer}_x']
        current_LUT = np.zeros((2 * (self.Qmax_w + 1), np.size(X)))
        indices = np.arange(-(self.Qmax_w + 1), self.Qmax_w + 1)  # Generate indices from negative to positive
        for i, iQ in enumerate(indices):
            current_LUT[i, :] = np.round(np.cos((i + 1) / curConvS_w * X) * curConvS_wx)
        self.QX_LUT.append(current_LUT)


    def _process_activations(self, iLayer):

        """Processes quantization parameters related to activations."""
        A_max = np.max(np.abs(self.conv_para[f'conv{iLayer}_A_merged']).flatten())
        curConvS = self.Qmax_A / A_max
        self.Q_params[f'conv{iLayer}_A_q'] = np.round(self.conv_para[f'conv{iLayer}_A_merged'] * curConvS)
        self.conv_para[f'conv{iLayer}_Qscale_A'] = curConvS

        # Q_act
        if iLayer == 0:
            self.QAct_para['input_activation_Qscale'] = self.Qmax_Act / np.max(self.features_dict['input_data'])
        nRBin = (self.Qmax_Act + 1) * 16
        if nRBin >= 8192:
            nRBin = 8192
        startSearchPoint = (self.Qmax_Act + 1) * 2
        FindThreshold.create_KL_layer(self.features_dict, iLayer, self.Qmax_Act, nRBin, startSearchPoint, self.ifold)

        # save QAct_para
        self.QAct_para[f'conv{iLayer}_activation_threshold'] = FindThreshold.KL_Layers[f'KL_Layer_{iLayer}'][
            'minKL_threshold']
        self.QAct_para[f'conv{iLayer}_activation_minKLValue'] = FindThreshold.KL_Layers[f'KL_Layer_{iLayer}'][
            'minKLValue']
        self.QAct_para[f'conv{iLayer}_activation_Qscale'] = FindThreshold.KL_Layers[f'KL_Layer_{iLayer}'][
            'minKL_QScale']
        self.QAct_para[f'conv{iLayer}_activation_KLEntropy'] = FindThreshold.KL_Layers[f'KL_Layer_{iLayer}'][
            'KLEntropy']
        self.QAct_para[f'conv{iLayer}_activation_Qmax'] = self.Qmax_Act
        self.QAct_para['unit_scale.activation_Qscale'] = 1
        if iLayer == 0:
            self.QAct_para['unit_scale.activation_Qmax'] = self.Qmax_Act

    def _calculate_final_parameters(self, iLayer):
        """Calculates the final quantization parameters."""

        # Calculate different parameters based on the layer's position
        if iLayer == 0:  # First layer
            denominator = (self.conv_para[f'conv{iLayer}_Qscale_A'] *
                           self.Q_params[f'conv{iLayer}_Qscale_wx'] *
                           self.QAct_para['input_activation_Qscale'])
            # Avoid division by zero, ensure the denominator is a scalar
            denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
            M_val = np.round(
                2 ** self.nBit_shift * self.QAct_para[f'conv{iLayer}_activation_Qscale'][0] / denominator)
            self.Q_params[f'M{iLayer}'] = M_val
            B_val = np.round(2 ** self.nBit_shift * self.QAct_para[f'conv{iLayer}_activation_Qscale'][0] *
                             self.Q_params[f'conv{iLayer}_convBias'])
            self.Q_params[f'B{iLayer}'] = np.round(B_val).flatten()  # Keep as a 1D array
            # print()
        elif iLayer == self.num_layers - 1:  # Last layer
            denominator = (self.conv_para[f'conv{iLayer}_Qscale_A'] *
                           self.Q_params[f'conv{iLayer}_Qscale_wx'] *
                           self.QAct_para[f'conv{iLayer - 1}_activation_Qscale'])
            # Avoid division by zero
            denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
            M_val = np.round(
                2 ** self.nBit_shift * self.QAct_para[f'conv{iLayer}_activation_Qscale'] / denominator)
            self.Q_params[f'M{iLayer}'] = float(np.mean(np.round(M_val)))  # Take the mean to ensure it is a scalar
            B_val = np.round(
                2 ** self.nBit_shift * self.QAct_para[f'conv{iLayer}_activation_Qscale'] * self.Q_params[f'conv{iLayer}_convBias'])
            self.Q_params[f'B{iLayer}'] = np.round(B_val).flatten()  # Keep as a 1D array
            # print()
            # Ensure it is a scalar value
            M = 1 / self.QAct_para[f'conv{iLayer}_activation_Qscale']  # Use directly
            B = np.array([0])  # Use the same zero bias for all channels
            self.Q_params[f'M{iLayer + 1}'] = float(M)  # Ensure it is a scalar
            self.Q_params[f'B{iLayer + 1}'] = B  # Zero bias
            # print()
        else:  # Intermediate layer
            denominator = (self.conv_para[f'conv{iLayer}_Qscale_A'] *
                           self.Q_params[f'conv{iLayer}_Qscale_wx'] *
                           self.QAct_para[f'conv{iLayer - 1}_activation_Qscale'])
            # Avoid division by zero
            denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
            M_val = np.round(
                2 ** self.nBit_shift * self.QAct_para[f'conv{iLayer}_activation_Qscale'] / denominator)
            self.Q_params[f'M{iLayer}'] = float(np.mean(np.round(M_val)))  # Take the mean to ensure it is a scalar
            B_val = np.round(
                2 ** self.nBit_shift * self.QAct_para[f'conv{iLayer}_activation_Qscale'] * self.Q_params[f'conv{iLayer}_convBias'])
            self.Q_params[f'B{iLayer}'] = np.round(B_val).flatten()  # Keep as a 1D array


    def _calculate_first_layer_params(self, iLayer):
        """Calculates parameters for the first layer."""

        denominator = (self.conv_para[f'conv{iLayer}_Qscale_A'] *
                       self.Q_params[f'conv{iLayer}_Qscale_wx'] *
                       self.QAct_para['input_activation_Qscale'])
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        M_val = np.round(
            2 ** self.nBit_shift * self.QAct_para[f'conv{iLayer}_activation_Qscale'][0] / denominator)
        self.Q_params[f'M{iLayer}'] = M_val
        B_val = np.round(2 ** self.nBit_shift * self.QAct_para[f'conv{iLayer}_activation_Qscale'][0] *
                         self.Q_params[f'conv{iLayer}_convBias'])
        self.Q_params[f'B{iLayer}'] = np.round(B_val).flatten()

    def quantize_layer(self, iLayer):
        """Quantizes the parameters of a single network layer."""

        # Get base parameters
        self.conv_para[f'conv{iLayer}_A'] = self.parameters_dict[f'conv{iLayer}_A']
        self.conv_para[f'conv{iLayer}_w'] = np.abs(self.parameters_dict[f'conv{iLayer}_w'])
        self.conv_para[f'conv{iLayer}_x'] = self.parameters_dict[f'conv{iLayer}_x']

        # # Get quantization bit-widths
        # [nBit_w, nBit_wx, nBit_A, nBit_Act, nBit_shift] = self._get_quantization_bits(iLayer)
        # Qmax_values = [getQ(bit) for bit in [nBit_w, nBit_wx, nBit_A, nBit_Act, nBit_shift]]

        # Execute each step
        self._merge_bn_layer(iLayer)
        self._generate_cosine_lut(iLayer)
        self._process_activations(iLayer)
        self._calculate_final_parameters(iLayer)

        # Save all necessary parameters to Q_params
        self.Q_params[f'conv{iLayer}_Qmax_w'] = self.Qmax_w
        self.Q_params[f'conv{iLayer}_Qmax_wx'] = self.Qmax_wx
        self.Q_params[f'conv{iLayer}_Qmax_A'] = self.Qmax_A
        self.Q_params[f'conv{iLayer}_Qscale_A'] = self.Q_params[f'conv{iLayer}_A_q']
        self.Q_params[f'conv{iLayer}_convBias'] = self.conv_para[f'conv{iLayer}_merged_bias']

        print(f'Processing layer{iLayer + 1}')

    def quantize_network(self):
        """Quantizes the entire network."""
        for iLayer in range(self.num_layers):
            self.quantize_layer(iLayer)
        return self.conv_para, self.Q_params, self.QAct_para, self.QX_LUT
