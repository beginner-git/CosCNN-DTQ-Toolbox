import os
import sys

from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.model import CosCNN, GroupCosCNN
from quantization.feature_extractor import FeatureParameterExtractor
from data_utils import load_and_preprocess_data



import re


def infer_groupcoscnn_arch_from_state_dict(state_dict, in_channels):
    """Infer (per_group_num_filters_list, groups_list) for GroupCosCNN from a checkpoint state_dict.

    We infer per-layer groups by using the shapes of CosConvLayer parameters:
      - feature_extractor.conv{i}.A has shape [out_channels_total, in_channels_per_group]

    Given total input channels of the layer (known from previous layer out_channels_total, and for layer0 = in_channels),
    groups_i = in_channels_total // in_channels_per_group.

    Returns:
        tuple(list[int], list[int]): (per_group_filters_list, groups_list)
    """
    # Find conv indices
    indices = set()
    for k in state_dict.keys():
        m = re.match(r"^feature_extractor\.conv(\d+)\.A$", k)
        if m:
            indices.add(int(m.group(1)))
    if not indices:
        raise ValueError("Cannot infer groups_list: no keys like 'feature_extractor.conv{i}.A' found in state_dict.")

    conv_indices = sorted(indices)

    per_group_filters_list = []
    groups_list = []

    in_channels_total = int(in_channels)

    for i in conv_indices:
        key = f"feature_extractor.conv{i}.A"
        A = state_dict[key]
        if hasattr(A, "shape"):
            out_channels_total, in_channels_per_group = int(A.shape[0]), int(A.shape[1])
        else:
            out_channels_total, in_channels_per_group = int(np.shape(A)[0]), int(np.shape(A)[1])

        if in_channels_per_group <= 0:
            raise ValueError(f"Invalid in_channels_per_group inferred at layer {i}: {in_channels_per_group}")

        if in_channels_total % in_channels_per_group != 0:
            raise ValueError(
                f"Cannot infer groups at layer {i}: in_channels_total={in_channels_total} "
                f"is not divisible by in_channels_per_group={in_channels_per_group}"
            )

        g = in_channels_total // in_channels_per_group
        if g < 1:
            raise ValueError(f"Inferred invalid groups at layer {i}: {g}")

        if out_channels_total % g != 0:
            raise ValueError(
                f"Cannot infer per-group filters at layer {i}: out_channels_total={out_channels_total} "
                f"is not divisible by groups={g}"
            )

        per_group_filters = out_channels_total // g

        groups_list.append(int(g))
        per_group_filters_list.append(int(per_group_filters))

        # Next layer total input channels = this layer total output channels
        in_channels_total = out_channels_total

    return per_group_filters_list, groups_list


def safe_load_groupcoscnn_state_dict(model, checkpoint_state_dict, in_channels, verbose=True):
    """Try strict load; if it fails due to shape mismatch, infer groups/filters and rebuild caller should handle."""
    try:
        model.load_state_dict(checkpoint_state_dict)
        return True, None
    except RuntimeError as e:
        if verbose:
            print("[WARN] Failed to load state_dict with provided config. Will try to infer per-layer groups from state_dict.")
            print("[WARN] Original error:", str(e))
        return False, e
def extract_features_and_parameters(ifold, model_path='../best_model/best_model.pth', in_channels=1):
    """
    Extract network features and parameters.
    """
    # Load and preprocess data using the function from data_utils
    XTrain, YTrain, XVal, YVal, XTest, YTest = load_and_preprocess_data(fold_number=ifold)

    # Load the model
    checkpoint = torch.load(model_path, weights_only=False)
    model_config = checkpoint['config']

    model_type = model_config.get('model_type', 'default')
    if model_type == 'group':
        # Try to build from config first; if config is missing/wrong, infer architecture from state_dict.
        cfg_num_filters_list = model_config.get('num_filters_list', None)
        cfg_groups_list = model_config.get('groups_list', None)

        if cfg_num_filters_list is None:
            # Fallback: infer per-layer (per-group filters, groups) directly
            inferred_num_filters_list, inferred_groups_list = infer_groupcoscnn_arch_from_state_dict(
                checkpoint['state_dict'], in_channels
            )
            cfg_num_filters_list = inferred_num_filters_list
            cfg_groups_list = inferred_groups_list

        model = GroupCosCNN(
            input_length=model_config['input_length'],
            in_channels=in_channels,
            num_classes=model_config['num_classes'],
            filter_length=model_config['filter_length'],
            num_filters_list=cfg_num_filters_list,
            groups_list=cfg_groups_list
        )

        ok, _err = safe_load_groupcoscnn_state_dict(model, checkpoint['state_dict'], in_channels)
        if not ok:
            inferred_num_filters_list, inferred_groups_list = infer_groupcoscnn_arch_from_state_dict(
                checkpoint['state_dict'], in_channels
            )
            print(f"[INFO] Inferred groups_list: {inferred_groups_list}")
            print(f"[INFO] Inferred per-group num_filters_list: {inferred_num_filters_list}")
            model = GroupCosCNN(
                input_length=model_config['input_length'],
                in_channels=in_channels,
                num_classes=model_config['num_classes'],
                filter_length=model_config['filter_length'],
                num_filters_list=inferred_num_filters_list,
                groups_list=inferred_groups_list
            )
            model.load_state_dict(checkpoint['state_dict'])
            # Update model_config so downstream code can use the inferred architecture if needed.
            model_config['num_filters_list'] = inferred_num_filters_list
            model_config['groups_list'] = inferred_groups_list
    elif model_type == 'default' or model_type == 'coscnn':
        model = CosCNN(
            input_length=model_config['input_length'],
            in_channels=in_channels,
            num_classes=model_config['num_classes'],
            filter_length=model_config['filter_length'],
            num_filters_list=model_config['num_filters_list']
        )
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f"Quantization only supports model_type in {{'default', 'group'}}, got: {model_type}")
    model.eval()

    # Create the feature extractor
    extractor = FeatureParameterExtractor(model)
    extractor.eval()

    # Prepare calibration data
    XCal = np.concatenate([XTrain, XVal], axis=0)

    # Support multi-channel input
    if len(XCal.shape) == 2:
        # Data is (num_samples, total_length), need to reshape
        calibration_data = torch.tensor(
            XCal.reshape(-1, in_channels, model_config['input_length']),
            dtype=torch.float32
        )
    elif len(XCal.shape) == 3:
        # Data is already (num_samples, channels, length), use directly
        calibration_data = torch.tensor(XCal, dtype=torch.float32)
    else:
        raise ValueError(f"Unexpected data shape: {XCal.shape}")

    # Extract features
    with torch.no_grad():
        _ = extractor(calibration_data)

    # Prepare data for return
    features_dict = {
        'input_data': calibration_data.numpy(),
        **extractor.features,
        'labels': np.concatenate([YTrain, YVal])
    }

    return features_dict, extractor.parameters_dict


def validate_quantized_model(quantized_model, ifold, in_channels=1, sig_path='../train/SIG.mat'):
    """
    Validate the performance of the quantized model.

    Args:
        quantized_model: The quantized model to validate
        ifold: Fold number
        in_channels: Number of input channels (default: 1)
        sig_path: Path to signal data (deprecated, kept for compatibility)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    quantized_model.to(device)

    # Load and preprocess data using the function from data_utils
    XTrain, YTrain, XVal, YVal, XTest, YTest = load_and_preprocess_data(fold_number=ifold)

    def getResult(model, XTest, YTest):
        model.eval()
        batch_size = 10000

        # Support multi-channel input
        if len(XTest.shape) == 2:
            # Data is (num_samples, total_length), need to reshape
            input_length = XTest.shape[1] // in_channels
            XTest = XTest.reshape(-1, in_channels, input_length)
        # If already 3D, keep as is

        test_dataset = TensorDataset(
            torch.tensor(XTest, dtype=torch.float32),
            torch.tensor(YTest, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        all_scores = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                scores = outputs.cpu().numpy()
                all_scores.extend(scores)
                all_labels.extend(labels.numpy())

        predictedScore = np.array(all_scores)
        predictedLabel = np.argmax(predictedScore, axis=1)
        actualLabel = np.array(all_labels)

        confmatrix = confusion_matrix(actualLabel, predictedLabel)
        accuracy = np.sum(np.eye(confmatrix.shape[0]) * confmatrix) / np.sum(confmatrix)

        return accuracy, confmatrix, predictedScore

    print("Evaluating the performance of the quantized model...")
    accuracy_Q, confmatrix_Q, predictedScore_Q = getResult(quantized_model, XTest, YTest)

    print(f"Quantized Model Accuracy: {accuracy_Q * 100:.2f}%")
    print("Quantized Model Confusion Matrix:")
    print(confmatrix_Q)

    return {
        'Accuracy_Q': accuracy_Q,
        'confmatrix_Q': confmatrix_Q,
        'predictedScore_Q': predictedScore_Q
    }


def validate_after_quantization(quantized_model, ifold, in_channels=1):
    """
    Validate model performance after quantization

    Args:
        quantized_model: The quantized model to validate
        ifold: Fold number
        in_channels: Number of input channels (default: 1)
    """
    print("Starting validation of the quantized model...")
    results = validate_quantized_model(quantized_model, ifold, in_channels=in_channels)

    accuracy = results['Accuracy_Q']

    return accuracy


def print_quantization_results(folds_accuracy):
    """
    Prints the validation results of the quantized model, supporting dynamic folds.

    Args:
        folds_accuracy (np.ndarray or list): The quantization accuracy for each fold.
    """
    num_folds = len(folds_accuracy)
    print("\n" + "=" * 50)
    print(f"Quantized Model Validation Results (Total Folds: {num_folds})")
    print("=" * 50)
    print(f"{'Fold':^10}{'Accuracy':^15}")
    print("-" * 25)

    for i, acc in enumerate(folds_accuracy, 1):
        print(f"{i:^10}{acc * 100:^15.2f}%")

    print("-" * 25)
    print(f"{'Average':^10}{np.mean(folds_accuracy) * 100:^15.2f}%")
    print(f"{'Std'    :^10}{np.std(folds_accuracy) * 100:^15.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    sample_folds = [0.85, 0.87, 0.88, 0.86, 0.89]
    print_quantization_results(sample_folds)