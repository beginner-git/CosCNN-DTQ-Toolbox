from dataclasses import dataclass
from typing import List
from copy import deepcopy


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    input_length: int = 4096
    in_channels = 1
    filter_length: int = 5
    num_filters_list: List[int] = None
    max_epochs: int = 240
    batch_size: int = 512
    # lr_start: float = 2e-4
    lr_end: float = 2e-5
    lr_drop_period: int = 50
    num_folds: int = 10
    model_type: str = "default"  # New: model type ("default" or "custom")
    custom_model_path: str = None  # New: custom model path
    perform_quantization = False
    quantization_script_path = "../quantization/main.py"
    # Added data paths
    sig_data_path: str = "../data/sigData.json"
    label_data_path: str = "../data/labelData.json"
    # Existing fields (abbreviated)
    # weight_decay: float = 1e-3  # Global weight decay
    enable_regularization: bool = True  # Enable regularization
    lr_start: float = 2e-4  # Base learning rate

    # New fields for A and w parameters
    weight_decay_A: float = 1e-3  # Weight decay for A
    weight_decay_w: float = 1e-3  # Weight decay for w
    lr_factor_A: float = 1.0  # Learning rate factor for A
    lr_factor_w: float = 1.0  # Learning rate factor for w

    def __post_init__(self):
        if self.num_filters_list is None:
            self.num_filters_list = [4, 8, 16, 32, 64]
        self.lr_drop_factor = (self.lr_end / self.lr_start) ** (1 / (self.max_epochs / self.lr_drop_period))

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid configuration key: {key}")

    def clone(self):
        new_config = TrainingConfig()
        for key, value in self.__dict__.items():
            setattr(new_config, key, value)
        return new_config
