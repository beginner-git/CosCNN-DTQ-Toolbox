import json
import os
import numpy as np
from scipy.io import loadmat

def load_and_preprocess_data(fold_number, data_dir="../data"):
    """
    Loads training, validation, and test data from a JSON file into NumPy arrays.

    Args:
        fold_number: The fold number to load.
        data_dir: The path to the data directory.

    Returns:
        A tuple containing XTrain, YTrain, XVal, YVal, XTest, YTest as NumPy arrays.
    """
    # Construct the file path
    file_path = os.path.join(data_dir, f"fold_{fold_number}_data.json")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load data from the JSON file
    with open(file_path, 'r') as f:
        data_dict = json.load(f)

    # Convert lists to NumPy arrays
    input_length = len(data_dict["XTrain"][0][0])  # Get the input length

    data_numpy = {
        "ifold": data_dict["iFold"],
        "XTrain": np.array(data_dict["XTrain"]),
        "YTrain": np.array(data_dict["YTrain"]),
        "XVal": np.array(data_dict["XVal"]),
        "YVal": np.array(data_dict["YVal"]),
        "XTest": np.array(data_dict["XTest"]),
        "YTest": np.array(data_dict["YTest"])
    }
    XTrain = data_numpy["XTrain"]
    YTrain = data_numpy["YTrain"]
    XVal = data_numpy["XVal"]
    YVal = data_numpy["YVal"]
    XTest = data_numpy["XTest"]
    YTest = data_numpy["YTest"]
    # ifold = data_numpy["ifold"]

    return XTrain, YTrain, XVal, YVal, XTest, YTest
