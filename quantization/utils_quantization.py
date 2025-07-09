import os
import sys

from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.model import CosCNN
from quantization.feature_extractor import FeatureParameterExtractor
from data_utils import load_and_preprocess_data


def extract_features_and_parameters(ifold, model_path='../best_model/best_model.pth', in_channels=1):
    """
    Extract network features and parameters.
    """
    # Load and preprocess data using the function from data_utils
    XTrain, YTrain, XVal, YVal, XTest, YTest = load_and_preprocess_data(fold_number=ifold)

    # Load the model
    checkpoint = torch.load(model_path)
    model_config = checkpoint['config']
    model = CosCNN(
        input_length=model_config['input_length'],
        in_channels=in_channels,
        num_classes=model_config['num_classes'],
        filter_length=model_config['filter_length'],
        num_filters_list=model_config['num_filters_list']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Create the feature extractor
    extractor = FeatureParameterExtractor(model)
    extractor.eval()

    # Prepare calibration data
    XCal = np.concatenate([XTrain, XVal], axis=0)
    calibration_data = torch.tensor(
        XCal.reshape(-1, 1, model_config['input_length']),
        dtype=torch.float32
    )

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


def validate_quantized_model(quantized_model, ifold, sig_path='../train/SIG.mat'):
    """
    Validate the performance of the quantized model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    quantized_model.to(device)

    # Load and preprocess data using the function from data_utils
    XTrain, YTrain, XVal, YVal, XTest, YTest = load_and_preprocess_data(fold_number=ifold)

    # # Label remapping: ensure labels are 0, 1, 2
    # label_map = {2: 0, 4: 1, 5: 2}  # Class B -> 0, Class D -> 1, Class E -> 2
    # YTest = np.array([label_map[x] for x in YTest])

    def getResult(model, XTest, YTest):
        model.eval()
        batch_size = 10000
        if len(XTest.shape) == 2:
            XTest = XTest.reshape(-1, 1, 4096)

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


def validate_after_quantization(quantized_model, ifold):
    """
    Validate model performance after quantization
    """
    print("Starting validation of the quantized model...")
    results = validate_quantized_model(quantized_model, ifold)

    accuracy = results['Accuracy_Q']

    return accuracy


def print_quantization_results(folds_accuracy):
    """
    Prints the validation results of the quantized model, supporting dynamic folds.

    Args:
        folds_accuracy (np.ndarray or list): The quantization accuracy for each fold.
    """
    num_folds = len(folds_accuracy)  # Dynamically get the number of folds
    print("\n" + "=" * 50)
    print(f"Quantized Model Validation Results (Total Folds: {num_folds})")
    print("=" * 50)
    print(f"{'Fold':^10}{'Accuracy':^15}")
    print("-" * 25)

    # Dynamically print the accuracy for each fold
    for i, acc in enumerate(folds_accuracy, 1):
        print(f"{i:^10}{acc * 100:^15.2f}%")

    print("-" * 25)
    # Calculate and print the average and standard deviation
    print(f"{'Average':^10}{np.mean(folds_accuracy) * 100:^15.2f}%")
    print(f"{'Std'    :^10}{np.std(folds_accuracy) * 100:^15.2f}%")
    print("=" * 50)

# Example call
if __name__ == "__main__":
    # Assume accuracy data for 5 folds
    sample_folds = [0.85, 0.87, 0.88, 0.86, 0.89]
    print_quantization_results(sample_folds)
