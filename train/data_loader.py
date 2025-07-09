import json
import os
from typing import Dict, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def main(export_to_json: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses signal data, with an option to export to JSON.

    Args:
    export_to_json: Whether to export the data to JSON files.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing signal data (shape (num_data, 1, data_length)) and label data.
    """
    # Define sig_path to be SIG.mat in the data folder of the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    sig_path = os.path.join(data_dir, "SIG.mat")

    print("Processing data from source file...")
    data_sig = loadmat(sig_path)
    SIG = data_sig['SIG']

    # Extract specified classes
    indices_B = np.arange(100, 200)
    indices_D_E = np.arange(300, 500)
    indices = np.concatenate((indices_B, indices_D_E))

    sigData = SIG[indices, :-2]
    labelData = SIG[indices, -1].flatten()

    # Reshape sigData to add a channel dimension
    sigData = np.expand_dims(sigData, axis=1)  # (num_data, data_length) -> (num_data, 1, data_length)

    # Center the data
    sigData = sigData - 0.5

    # Encode labels
    le = LabelEncoder()
    labelData = le.fit_transform(labelData)

    # Export to JSON (if required)
    if export_to_json:
        export_data_to_json(sigData, labelData)

    print(f"Shape of sigData: {sigData.shape}") # Add print to confirm shapes
    print(f"Shape of labelData: {labelData.shape}")

    return sigData, labelData



def export_data_to_json(sigData: np.ndarray, labelData: np.ndarray) -> Dict[str, str]:
    """
    Exports data to JSON files.

    Args:
    sigData: Signal data (should be shape (num_data, 1, data_length)).
    labelData: Label data.

    Returns:
    Dict[str, str]: A dictionary containing the paths to the JSON files.
    """
    file_paths = {}

    # Define the export directory as the 'data' folder in the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_export_dir = os.path.join(os.path.dirname(current_dir), "data")

    # Ensure the directory exists
    os.makedirs(data_export_dir, exist_ok=True)

    # Export sigData
    sig_data_path = os.path.join(data_export_dir, 'sigData.json')
    with open(sig_data_path, 'w') as f:
        # tolist() can handle multi-dimensional arrays
        json.dump(sigData.tolist(), f)
    file_paths['sigData'] = sig_data_path

    # Export labelData
    label_data_path = os.path.join(data_export_dir, 'labelData.json')
    with open(label_data_path, 'w') as f:
        json.dump(labelData.tolist(), f)
    file_paths['labelData'] = label_data_path

    print(f"Data exported to {data_export_dir}")
    return file_paths


if __name__ == '__main__':
    main()
