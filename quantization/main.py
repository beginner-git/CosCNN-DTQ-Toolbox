import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.QuantizationModel import QuantizedCosCNN
from utils_quantization import extract_features_and_parameters, print_quantization_results, validate_after_quantization, infer_groupcoscnn_arch_from_state_dict
from ui import ModernInputDialog, InitialChoiceDialog
from quantizer import NetworkQuantizer


def to_python_native(obj):
    """
    Recursively convert numpy types to Python native types.
    This is important for torch.save/load and JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: to_python_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_native(v) for v in obj]
    return obj


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
    try:
        current_script_path = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_script_path)
    except NameError:
        current_dir = os.getcwd()
        print("Warning: Could not determine script path via __file__. Using current working directory.")

    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')

    cv_indices_path = os.path.join(data_dir, 'cv_indices.json')
    in_channels_path = os.path.join(data_dir, 'in_channels.json')

    loaded_file = ""
    try:
        loaded_file = cv_indices_path
        print(f"Attempting to load: {cv_indices_path}")
        with open(cv_indices_path, 'r') as f:
            cv_indices = json.load(f)
            if not isinstance(cv_indices, list):
                print(f"Error: The content of {cv_indices_path} is not a list.")
                return None, None
        print(f"Successfully loaded cv_indices.")

        loaded_file = in_channels_path
        print(f"Attempting to load: {in_channels_path}")
        with open(in_channels_path, 'r') as f:
            loaded_value = json.load(f)
            if isinstance(loaded_value, int):
                in_channels = loaded_value
                print(f"Successfully loaded in_channels: {in_channels}")
            else:
                print(
                    f"Error: The content of file {in_channels_path} is not a valid integer. Actual type: {type(loaded_value)}")
                return None, None

    except FileNotFoundError as e:
        print(f"Error: File not found - {loaded_file or getattr(e, 'filename', 'N/A')}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file - {loaded_file} - {e}")
        return None, None
    except Exception as e:
        print(f"An unknown error occurred while reading file {loaded_file}: {type(e).__name__}: {e}")
        return None, None

    return cv_indices, in_channels


def get_num_classes_from_checkpoint(checkpoint):
    """
    Infer num_classes from the checkpoint.
    First try config, then try classifier weight shape.
    """
    model_config = checkpoint.get('config', {})
    num_classes = model_config.get('num_classes', None)

    if num_classes is not None:
        return int(num_classes)

    state_dict = checkpoint.get('state_dict', checkpoint)
    for key in ['classifier.1.weight', 'classifier.weight', 'fc.weight']:
        if key in state_dict and state_dict[key].ndim == 2:
            return int(state_dict[key].shape[0])

    raise ValueError("Could not infer num_classes from checkpoint. Please specify it manually.")


def process_fold(ifold, num_folds, w_max, nBit_params=None, in_channels=1):
    """Processes the quantization and validation logic for a single fold and exports the quantized model"""
    model_path = f'../trained_models/fold_{ifold + 1}/model_fold_{ifold + 1}.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, weights_only=False)
    model_config = checkpoint.get('config', {})
    model_type = model_config.get('model_type', 'default')
    # For GroupCosCNN, the checkpoint state_dict is the source of truth. Infer per-layer groups to avoid config mismatch.
    if model_type == 'group':
        try:
            _inferred_num_filters, _inferred_groups = infer_groupcoscnn_arch_from_state_dict(checkpoint['state_dict'], in_channels)
            groups_list = _inferred_groups
            # Optional: keep config consistent for later use
            model_config['groups_list'] = groups_list
        except Exception as e:
            print(f"[WARN] Could not infer groups_list from state_dict, falling back to config. Error: {e}")
            groups_list = model_config.get('groups_list', None)
    else:
        groups_list = None
    groups = in_channels if model_type == 'group' else 1

    num_classes = get_num_classes_from_checkpoint(checkpoint)
    print(f"Inferred num_classes from original model: {num_classes}")

    features_dict, parameters_dict = extract_features_and_parameters(ifold + 1, model_path, in_channels)

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

    quantizer = NetworkQuantizer(parameters_dict, nBit_params, features_dict, num_layers, w_max, ifold)
    conv_para, Q_params, QAct_para, QX_LUT = quantizer.quantize_network()
    Qmax_Act = QAct_para['unit_scale.activation_Qmax']

    initial_M = float(np.round(2 ** nBit_shift * QAct_para['input_activation_Qscale'] /
                               QAct_para['unit_scale.activation_Qscale']))

    initial_B = np.zeros(in_channels)

    num_channels = [len(features_dict['input_data'][0])] + [len(features_dict[f'conv{i}'][0]) for i in
                                                            range(num_layers - 1)]
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

        M_val = Q_params[f'M{iLayer}']
        if isinstance(M_val, np.ndarray):
            M_val = float(np.mean(M_val))
        M_list.append(float(M_val))

        B_val = Q_params[f'B{iLayer}']
        if isinstance(B_val, np.ndarray):
            B_list.append(B_val.flatten().tolist())
        else:
            B_list.append([float(B_val)])

        nBit_shift_list.append(int(nBit_shift))
        Qmax_list.append(int(QAct_para[f'conv{iLayer}_activation_Qmax']))

    input_length = len(features_dict['input_data'][0][0])
    filter_length = len(conv_para['conv0_x'][0][0])

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
        final_B=B_final,
        groups=groups,
        groups_list=groups_list
    )

    for module in quantized_model.classifier:
        if isinstance(module, nn.Linear):
            print("Loading classifier parameters...")
            module.weight.data = torch.tensor(parameters_dict['fc_weight'])
            module.bias.data = torch.tensor(parameters_dict['fc_bias'])
            break

    save_dir = 'quantized_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'quantized_model_fold_{ifold + 1}.pth')

    quant_config = to_python_native({
        'input_length': int(input_length),
        'num_classes': int(num_classes),
        'filter_length': int(filter_length),
        'num_filters_list': [int(x) for x in num_filters_list],
        'num_channels': [int(x) for x in num_channels],
        'M_list': [float(x) for x in M_list],
        'B_list': B_list,
        'nBit_shift_list': [int(x) for x in nBit_shift_list],
        'Qmax_list': [int(x) for x in Qmax_list],
        'initial_M': float(initial_M),
        'initial_B': initial_B.tolist(),
        'initial_nBit_shift': int(nBit_shift),
        'initial_Qmax': float(Qmax_Act),
        'final_M': float(M_final) if not isinstance(M_final, np.ndarray) else float(np.mean(M_final)),
        'final_B': B_final.tolist() if isinstance(B_final, np.ndarray) else [float(B_final)],
        'model_type': str(model_type),
        'groups': int(groups),
        'groups_list': to_python_native(groups_list) if groups_list is not None else None,
    })

    checkpoint_to_save = {
        'state_dict': quantized_model.state_dict(),
        'config': quant_config
    }

    torch.save(checkpoint_to_save, save_path)
    print(f"Quantized model saved to {save_path}")

    print(f"\nSaved config summary:")
    print(f"  num_classes: {quant_config['num_classes']}")
    print(f"  groups: {quant_config['groups']}")
    print(f"  M_list: {quant_config['M_list']}")
    print(f"  nBit_shift_list: {quant_config['nBit_shift_list']}")
    print(f"  initial_M: {quant_config['initial_M']}")
    print(f"  initial_nBit_shift: {quant_config['initial_nBit_shift']}")

    accuracy = validate_after_quantization(quantized_model, ifold + 1, in_channels)
    return accuracy, nBit_params


def main():
    all_weights = []

    cv_indices, in_channels = load_cv_and_channels()

    num_folds = len(np.unique(cv_indices))
    folds_accuracy = np.empty(num_folds)

    for fold in range(1, num_folds + 1):
        try:
            fold_model_path = f'../trained_models/fold_{fold}/model_fold_{fold}.pth'
            checkpoint = torch.load(fold_model_path, weights_only=False)
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

    nBit_params = None
    for ifold in range(num_folds):
        accuracy, nBit_params = process_fold(ifold, num_folds, w_max, nBit_params, in_channels)
        folds_accuracy[ifold] = accuracy

    print_quantization_results(folds_accuracy)


if __name__ == '__main__':
    main()