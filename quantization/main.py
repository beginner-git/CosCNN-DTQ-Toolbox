import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.QuantizationModel import QuantizedCosCNN
from utils_quantization import extract_features_and_parameters, print_quantization_results, validate_after_quantization
from ui import ModernInputDialog, InitialChoiceDialog
from quantizer import NetworkQuantizer


def load_cv_and_channels():
    """
    Loads cv_indices (list) and in_channels (int) from the ../data directory.
    Assumes in_channels.json contains a single integer directly.

    Returns:
        tuple: A tuple containing cv_indices (list) and in_channels (int).
               Returns (None, None) if either file fails to load or if in_channels is not an integer.
    """
    cv_indices = None
    in_channels = None
    # Get the directory of the current script
    try:
        current_script_path = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_script_path)
    except NameError:
        current_dir = os.getcwd()
        print("Warning: Could not determine script path via __file__. Using current working directory.")

    # Get the parent directory
    parent_dir = os.path.dirname(current_dir)
    # Build the path to the 'data' folder
    data_dir = os.path.join(parent_dir, 'data')

    # Build file paths
    cv_indices_path = os.path.join(data_dir, 'cv_indices.json')
    in_channels_path = os.path.join(data_dir, 'in_channels.json')

    loaded_file = "" # Track which file is being loaded for error reporting
    try:
        # Read the cv_indices.json file
        loaded_file = cv_indices_path
        print(f"Attempting to load: {cv_indices_path}")
        with open(cv_indices_path, 'r') as f:
            cv_indices = json.load(f)
            # Add a check to ensure cv_indices is a list
            if not isinstance(cv_indices, list):
                 print(f"Error: The content of {cv_indices_path} is not a list.")
                 return None, None
        print(f"Successfully loaded cv_indices.")

        # --- Modification Start ---
        # Read the in_channels.json file (assuming it contains only an integer)
        loaded_file = in_channels_path
        print(f"Attempting to load: {in_channels_path}")
        with open(in_channels_path, 'r') as f:
            loaded_value = json.load(f)
            # Check if the loaded value is an integer
            if isinstance(loaded_value, int):
                in_channels = loaded_value
                print(f"Successfully loaded in_channels: {in_channels}")
            else:
                # If it's not an integer, print an error and return None
                print(f"Error: The content of file {in_channels_path} is not a valid integer. Actual type: {type(loaded_value)}")
                return None, None
        # --- Modification End ---

    except FileNotFoundError as e:
        print(f"Error: File not found - {loaded_file or getattr(e, 'filename', 'N/A')}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file - {loaded_file} - {e}")
        return None, None
    # ValueError no longer needs to be caught specifically, as isinstance handles the type check
    # except ValueError:
    #     print(f"错误: {in_channels_path} 中的 'in_channels' 值不是有效的整数。")
    #     return None, None
    except Exception as e:
        print(f"An unknown error occurred while reading file {loaded_file}: {type(e).__name__}: {e}")
        return None, None

    return cv_indices, in_channels

def process_fold(ifold, num_folds, w_max, nBit_params=None, in_channels=1):
    """Processes the quantization and validation logic for a single fold and exports the quantized model to the parent folder"""
    model_path = f'../trained_models/fold_{ifold + 1}/model_fold_{ifold + 1}.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Extract features and parameters
    features_dict, parameters_dict = extract_features_and_parameters(ifold + 1, model_path, in_channels)

    # Start the quantization process
    num_layers = len(features_dict) - 2
    if ifold == 0 and nBit_params is None:
        nBit_params = {}
        dialog = InitialChoiceDialog()
        customize = dialog.get_choice()

        if customize:
            for iLayer in range(num_layers):
                nBit_params[f'conv{iLayer}_nBit_w'] = 12
                nBit_params[f'conv{iLayer}_nBit_shift'] = 23
                layer_dialog = ModernInputDialog(iLayer)
                (nBit_params[f'conv{iLayer}_nBit_wx'], nBit_params[f'conv{iLayer}_nBit_A'],
                 nBit_params[f'conv{iLayer}_nBit_Act']) = layer_dialog.get_values()
        else:
            for iLayer in range(num_layers):
                nBit_params[f'conv{iLayer}_nBit_w'] = 12
                nBit_params[f'conv{iLayer}_nBit_wx'] = 8
                nBit_params[f'conv{iLayer}_nBit_A'] = 8
                nBit_params[f'conv{iLayer}_nBit_Act'] = 8
                nBit_params[f'conv{iLayer}_nBit_shift'] = 23

    print(f'Processing fold {ifold + 1}\n')
    nBit_shift = 23

    # Quantization logic
    quantizer = NetworkQuantizer(parameters_dict, nBit_params, features_dict, num_layers, w_max, ifold)
    conv_para, Q_params, QAct_para, QX_LUT = quantizer.quantize_network()
    Qmax_Act = QAct_para['unit_scale.activation_Qmax']

    initial_M = float(np.round(2 ** nBit_shift * QAct_para['input_activation_Qscale'] /
                              QAct_para['unit_scale.activation_Qscale']))
    initial_B = np.array([0])

    num_channels = [len(features_dict['input_data'][0])] + [len(features_dict[f'conv{i}'][0]) for i in range(num_layers - 1)]
    num_filters_list = [len(features_dict[f'conv{i}'][1]) for i in range(num_layers)]
    params_list = []
    M_list = []
    B_list = []
    nBit_shift_list = []
    Qmax_list = []

    M_final = Q_params[f'M{num_layers}']
    B_final = Q_params[f'B{num_layers}']

    for iLayer in range(num_layers):
        params = {
            'Qmax_w': Q_params[f'conv{iLayer}_Qmax_w'],
            'w': torch.tensor(Q_params[f'conv{iLayer}_w_q'], dtype=torch.long).unsqueeze(0).unsqueeze(0),
            'QX_LUT': torch.tensor(QX_LUT[iLayer], dtype=torch.float32),
            'A': torch.tensor(Q_params[f'conv{iLayer}_A_q'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        }
        params_list.append(params)
        M_list.append(Q_params[f'M{iLayer}'])
        B_list.append(Q_params[f'B{iLayer}'])
        nBit_shift_list.append(nBit_shift)
        Qmax_list.append(QAct_para[f'conv{iLayer}_activation_Qmax'])

    input_length = len(features_dict['input_data'][0][0])
    num_classes = 3
    filter_length = len(conv_para['conv0_x'][0][0])

    # Create the quantized model
    print("\nCreating quantized model...")
    quantized_model = QuantizedCosCNN(
        input_length=input_length,
        num_classes=num_classes,
        filter_length=filter_length,
        num_filters_list=num_filters_list,
        params_list=params_list,
        M_list=M_list,
        B_list=B_list,
        nBit_shift_list=nBit_shift_list,
        Qmax_list=Qmax_list,
        num_channels=num_channels,
        initial_M=initial_M,
        initial_B=initial_B,
        initial_nBit_shift=nBit_shift,
        initial_Qmax=Qmax_Act,
        final_M=M_final,
        final_B=B_final
    )

    # Load classifier parameters
    for module in quantized_model.classifier:
        if isinstance(module, nn.Linear):
            print("Loading classifier parameters...")
            module.weight.data = torch.tensor(parameters_dict['fc_weight'])
            module.bias.data = torch.tensor(parameters_dict['fc_bias'])
            break

    # **New feature: Export the quantized model to the parent folder**
    save_dir = 'quantized_models'  # Save directory in the parent folder
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it does not exist
    save_path = os.path.join(save_dir, f'quantized_model_fold_{ifold + 1}.pth')
    torch.save(quantized_model.state_dict(), save_path)  # Save the model state dictionary
    print(f"Quantized model for fold {ifold + 1} saved to {save_path}")

    # Validate the quantized model
    accuracy = validate_after_quantization(quantized_model, ifold + 1)
    return accuracy, nBit_params

def main():
    # Get the maximum weight from all folds
    all_weights = []

    cv_indices, in_channels = load_cv_and_channels()

    num_folds = len(np.unique(cv_indices))
    folds_accuracy = np.empty(num_folds)

    for fold in range(1, num_folds + 1):
        try:
            fold_model_path = f'../trained_models/fold_{fold}/model_fold_{fold}.pth'
            checkpoint = torch.load(fold_model_path)
            for key, value in checkpoint['state_dict'].items():
                if isinstance(value, torch.Tensor) and 'w' in key:
                    weights = value.cpu().numpy()
                    all_weights.append(np.abs(weights))
        except FileNotFoundError:
            print(f"Warning: Model for fold {fold} not found at {fold_model_path}")
            continue
        except Exception as e:
            print(f"Error loading fold {fold}: {str(e)}")
            continue

    w_max = np.max([np.max(w) for w in all_weights]) if all_weights else None
    if w_max is None:
        print("No weights found in any fold")
        return

    # Process each fold
    nBit_params = None
    for ifold in range(num_folds):
        accuracy, nBit_params = process_fold(ifold, num_folds, w_max, nBit_params, in_channels)
        folds_accuracy[ifold] = accuracy

    # Print the results
    print_quantization_results(folds_accuracy)

if __name__ == '__main__':
    main()