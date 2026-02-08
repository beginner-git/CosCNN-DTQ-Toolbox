import numpy as np
import json
import os
from typing import Tuple, Dict
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def stratified_k_fold(labels, k):
    """
    Randomly divides a set of labels into k folds, ensuring that the number of
    instances for each label is as equal as possible across the folds.

    Args:
    labels: A list of labels, e.g., [0, 1, 2, 3, 0, 1, 2...].
    k: The number of folds.

    Returns:
    fold_indices: The fold index (from 0 to k-1) for each sample.
    """
    unique_labels = np.unique(labels)
    label_indices = {}

    for label in unique_labels:
        label_indices[label] = np.where(np.array(labels) == label)[0]
        np.random.shuffle(label_indices[label])

    fold_indices = np.zeros(len(labels), dtype=int)

    for label, indices in label_indices.items():
        fold_sizes = np.ones(k, dtype=int) * (len(indices) // k)
        remainder = len(indices) % k
        if remainder > 0:
            fold_sizes[:remainder] += 1

        start_idx = 0
        for fold_idx in range(k):
            end_idx = start_idx + fold_sizes[fold_idx]
            fold_indices[indices[start_idx:end_idx]] = fold_idx
            start_idx = end_idx

    return fold_indices



class DataManager:
    """A class to handle data loading and preprocessing."""

    def __init__(self, sig_path: str, divide: int, data_export_dir: str = None):
        self.sig_path = sig_path  # Can be None
        self.divide = divide
        if data_export_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            self.data_export_dir = os.path.join(parent_dir, 'data')
        else:
            self.data_export_dir = data_export_dir
        os.makedirs(self.data_export_dir, exist_ok=True)

        # Ensure the directory exists
        os.makedirs(self.data_export_dir, exist_ok=True)

    def export_cv_indices_and_metadata(self, cv_indices: np.ndarray, num_classes: int) -> Dict[str, str]:
        """
        Exports cv_indices, num_classes, and num_folds to JSON files.

        Args:
        cv_indices: Cross-validation indices.
        num_classes: The number of classes.

        Returns:
        Dict[str, str]: A dictionary containing the paths to the exported files.
        """
        file_paths = {}

        # Export cv_indices
        cv_indices_path = os.path.join(self.data_export_dir, 'cv_indices.json')
        with open(cv_indices_path, 'w') as f:
            json.dump(cv_indices.tolist(), f)
        file_paths['cv_indices'] = cv_indices_path

        # Export num_classes
        metadata_path = os.path.join(self.data_export_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({'num_classes': int(num_classes)}, f)
        file_paths['metadata'] = metadata_path

        # Export divide as num_folds
        num_folds_path = os.path.join(self.data_export_dir, 'num_folds.json')
        with open(num_folds_path, 'w') as f:
            json.dump({'num_folds': self.divide}, f)
        file_paths['num_folds'] = num_folds_path

        print(f"cv_indices, num_classes, and num_folds have been exported to {self.data_export_dir}")
        return file_paths

    def load_data_from_json(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads sigData and labelData from JSON files.
        """
        sig_data_path = os.path.join(self.data_export_dir, 'sigData.json')
        label_data_path = os.path.join(self.data_export_dir, 'labelData.json')

        if not os.path.exists(sig_data_path) or not os.path.exists(label_data_path):
            raise FileNotFoundError(f"Required JSON files (sigData.json or labelData.json) not found in {self.data_export_dir}")

        with open(sig_data_path, 'r') as f:
            sigData = np.array(json.load(f))

        with open(label_data_path, 'r') as f:
            labelData = np.array(json.load(f))

        print(f"sigData and labelData loaded from {self.data_export_dir}")
        return sigData, labelData

    def load_and_preprocess_data(self, sig_data_path=None, label_data_path=None):
        """
        Load and preprocess signal data from JSON files, or raise an error if files are missing.

        Args:
            sig_data_path (str, optional): Path to signal data JSON file. Defaults to None.
            label_data_path (str, optional): Path to label data JSON file. Defaults to None.
        """
        # Use provided paths or default paths
        sig_json_path = sig_data_path or os.path.join(self.data_export_dir, "sigData.json")
        label_json_path = label_data_path or os.path.join(self.data_export_dir, "labelData.json")

        try:
            # Load data from JSON files
            with open(sig_json_path, 'r') as f:
                sigData = np.array(json.load(f))
            with open(label_json_path, 'r') as f:
                labelData = np.array(json.load(f))
        except FileNotFoundError:
            if self.sig_path is None:
                raise FileNotFoundError(
                    f"JSON files not found at {sig_json_path} and {label_json_path}, "
                    "and no original .mat file path (sig_path) was provided."
                )
            # If sig_path is provided, load from .mat file as a fallback (optional logic)
            print("JSON files not found. Loading from original .mat file...")
            data_sig = loadmat(self.sig_path)
            SIG = data_sig['SIG']
            indices_B = np.arange(100, 200)
            indices_D_E = np.arange(300, 500)
            indices = np.concatenate((indices_B, indices_D_E))
            sigData = SIG[indices, :-2]
            labelData = SIG[indices, -1].flatten()
            sigData = np.expand_dims(sigData, axis=1)
            sigData = sigData - 0.5
            # Optionally save to JSON for future use
            with open(sig_json_path, 'w') as f:
                json.dump(sigData.tolist(), f)
            with open(label_json_path, 'w') as f:
                json.dump(labelData.tolist(), f)

        # Additional preprocessing
        cv_indices = stratified_k_fold(labelData, self.divide)  # Assume this function exists
        cv_indices = cv_indices + 1

        le = LabelEncoder()
        labelData = le.fit_transform(labelData)
        num_classes = len(le.classes_)

        return sigData, labelData, cv_indices, num_classes

    def export_to_json(self, sigData, labelData):
        """
        Exports data to JSON files.
        """
        sig_data_path = os.path.join(self.data_export_dir, 'sigData.json')
        label_data_path = os.path.join(self.data_export_dir, 'labelData.json')
        with open(sig_data_path, 'w') as f:
            json.dump(sigData.tolist(), f)
        with open(label_data_path, 'w') as f:
            json.dump(labelData.tolist(), f)
        print(f"sigData and labelData have been exported to {self.data_export_dir}")

    @staticmethod
    def get_train_val_idx(y_train_val: np.ndarray, coeff: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """Splits data into training and validation sets."""
        class_labels = np.unique(y_train_val)
        itrain = []
        ival = []

        for cls in class_labels:
            cur_class_idx = np.where(y_train_val == cls)[0]
            n_train = int(round(coeff * len(cur_class_idx)))
            itrain.extend(cur_class_idx[:n_train])
            ival.extend(cur_class_idx[n_train:])

        return np.array(itrain), np.array(ival)
