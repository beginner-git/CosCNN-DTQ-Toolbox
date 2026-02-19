# CosCNN-DTQ Toolbox
# Copyright (C) 2024-2025 Yimin Liu
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

from dataclasses import dataclass
from typing import List, Optional
from copy import deepcopy


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    input_length: int = 4096
    in_channels = 1
    filter_length: int = 5
    num_filters_list: List[int] = None
    groups_list: Optional[List[Optional[int]]] = None  # Per-layer groups for GroupCosCNN
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
    def _sync_groups_list(self):
        """Ensure groups_list length matches num_filters_list length."""
        if self.num_filters_list is None:
            return
        n = len(self.num_filters_list)
        if getattr(self, 'groups_list', None) is None:
            self.groups_list = [None for _ in range(n)]
            return
        gl = list(self.groups_list)
        if len(gl) < n:
            gl += [None] * (n - len(gl))
        self.groups_list = gl[:n]

    def __post_init__(self):
        if self.num_filters_list is None:
            self.num_filters_list = [4, 8, 16, 32, 64]
        self._sync_groups_list()
        self.lr_drop_factor = (self.lr_end / self.lr_start) ** (1 / (self.max_epochs / self.lr_drop_period))

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid configuration key: {key}")

        # Keep groups_list aligned with num_filters_list
        self._sync_groups_list()

    def clone(self):
        new_config = TrainingConfig()
        for key, value in self.__dict__.items():
            setattr(new_config, key, value)
        return new_config
