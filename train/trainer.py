import torch.nn as nn
import numpy as np
from torch.optim import Adam
import copy
from typing import Dict, Any, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import os
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader, TensorDataset

from config import TrainingConfig


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device

    def train_model(self, model: nn.Module, train_loader: DataLoader,
                    val_loader: DataLoader, test_loader: DataLoader) -> Tuple[nn.Module, float, np.ndarray]:
        """Train the model and return results"""
        criterion = nn.CrossEntropyLoss()

        params_A = [param for name, param in model.named_parameters() if 'A' in name]
        params_w = [param for name, param in model.named_parameters() if 'w' in name]
        # params_other = [param for name, param in model.named_parameters() if 'A' not in name and 'w' not in name]
        # Define parameter groups for optimizer
        param_groups = [
            {'params': params_A,
             'weight_decay': self.config.weight_decay_A if self.config.enable_regularization else 0.0,
             'lr': self.config.lr_start * self.config.lr_factor_A},
            {'params': params_w,
             'weight_decay': self.config.weight_decay_w if self.config.enable_regularization else 0.0,
             'lr': self.config.lr_start * self.config.lr_factor_w},
            # {'params': params_other,
            #  'weight_decay': self.config.weight_decay if self.config.enable_regularization else 0.0,
            #  'lr': self.config.lr_start}
        ]

        # weight_decay = self.config.weight_decay if self.config.enable_regularization else 0.0
        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.lr_drop_period,
            gamma=self.config.lr_drop_factor
        )

        model.to(self.device)
        best_val_accuracy = 0.0
        best_model_state = copy.deepcopy(model.state_dict())

        for epoch in range(self.config.max_epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                val_accuracy = evaluate(model, val_loader, self.device)  # Call module-level function
                print(f'Epoch [{epoch + 1}/{self.config.max_epochs}], '
                      f'Loss: {running_loss / len(train_loader):.4f}, '
                      f'Val Accuracy: {val_accuracy * 100:.2f}%')

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_state)
        test_results = test(model, test_loader, self.device)  # Call module-level function

        return model, test_results['accuracy'], test_results['confusion_matrix']


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Evaluate model performance on the validation set

    Args:
        model: The model to be evaluated
        data_loader: Data loader
        device: Computing device (CPU/GPU)

    Returns:
        float: Accuracy on the validation set
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def test(model: nn.Module, test_loader: DataLoader, device: torch.device, class_names=None) -> Dict[str, Any]:
    """Comprehensively evaluate model performance on the test set

    Args:
        model: The model to be evaluated
        test_loader: Test data loader
        device: Computing device
        class_names: List of class names (optional)

    Returns:
        dict: A dictionary containing various evaluation metrics
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    accuracy = (all_labels == all_predictions).mean()
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(all_labels)))]

    class_report = classification_report(all_labels, all_predictions,
                                         target_names=class_names, digits=4)

    class_accuracies = {}
    for i, name in enumerate(class_names):
        mask = (all_labels == i)
        class_acc = (all_predictions[mask] == all_labels[mask]).mean() if mask.sum() > 0 else 0
        class_accuracies[name] = class_acc

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'class_accuracies': class_accuracies,
        'predictions': all_predictions,
        'probabilities': all_probabilities
    }