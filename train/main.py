import importlib.util
import json
import os
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
import threading
from queue import Queue
import numpy as np
from typing import Callable, Optional

from config import TrainingConfig
from data_manager import DataManager
from trainer import ModelTrainer
from models.model import CosCNN
from training_ui import TrainingConfigUI


class TrainingApplication:
    """
    Main application class for the Model Training System.
    Manages the GUI, configuration, and training process for a machine learning model.
    """

    def __init__(self):
        """
        Initialize the TrainingApplication.
        Sets up the Tkinter window, configuration, and UI components.
        """
        # Initialize the main Tkinter window
        self.root = tk.Tk()
        self.root.title("CosCNN-DTQ Toolbox")
        self.root.geometry("760x700")
        self.root.minsize(760, 640)

        # Attempt to set the window icon, silently fail if not found
        try:
            self.root.iconbitmap("icons/app_icon.ico")
        except:
            pass

        # Queue for thread-safe message passing between training thread and GUI
        self.message_queue = Queue()
        # Training configuration object
        self.config = TrainingConfig()

        # Configure visual styles for the UI
        self.configure_styles()

        # Create a paned window to split the UI into left and right sections
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_frame = ttk.Frame(self.paned_window)
        self.right_frame = ttk.Frame(self.paned_window)

        self.paned_window.add(self.left_frame, weight=1)
        self.paned_window.add(self.right_frame, weight=1)

        # Set the minimum width for the right frame
        self.min_right_frame_width = 365  # Set the required minimum width

        # Initialize UI components
        self.ui = TrainingConfigUI(self.left_frame, self.on_config_update)
        self.create_control_panel()
        self.create_log_area()

        # Configure grid weights for resizing
        self.left_frame.columnconfigure(0, weight=1)
        self.left_frame.rowconfigure(0, weight=1)

        # Training status flags and thread
        self.is_training = False
        self.training_thread = None

        # Function to ensure the right frame maintains its minimum width
        def check_sash_position(event=None):
            self.root.update_idletasks()
            paned_width = self.paned_window.winfo_width()
            current_sash = self.paned_window.sashpos(0)
            if current_sash > paned_width - self.min_right_frame_width:
                new_sash = paned_width - self.min_right_frame_width
                if new_sash > 0:
                    self.paned_window.sashpos(0, new_sash)

        self.root.bind("<Configure>", check_sash_position)
        self.paned_window.bind("<ButtonRelease-1>", check_sash_position)
        self.root.after(100, check_sash_position)

        # Initialize sash position after the window is displayed
        self.root.after(100, check_sash_position)

        # Function to ensure the right frame maintains its minimum width
        def check_sash_position(event=None):
            self.root.update_idletasks()
            paned_width = self.paned_window.winfo_width()
            current_sash = self.paned_window.sashpos(0)
            if current_sash > paned_width - self.min_right_frame_width:
                new_sash = paned_width - self.min_right_frame_width
                if new_sash > 0:
                    self.paned_window.sashpos(0, new_sash)

        self.root.bind("<Configure>", check_sash_position)
        self.paned_window.bind("<ButtonRelease-1>", check_sash_position)
        self.root.after(100, check_sash_position)

        # Initialize sash position after the window is displayed
        self.root.after(100, check_sash_position)

    def configure_styles(self):
        """
        Configure ttk styles for a modern UI appearance.
        Selects the best available theme based on the platform.
        """
        style = ttk.Style()
        available_themes = style.theme_names()
        # Prioritize themes for a modern look
        if 'vista' in available_themes:
            style.theme_use('vista')
        elif 'clearlooks' in available_themes:
            style.theme_use('clearlooks')
        elif 'aqua' in available_themes:
            style.theme_use('aqua')
        else:
            style.theme_use('clam')

        # Define styles for UI widgets
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TLabelframe", font=("Helvetica", 11, "bold"))
        style.configure("TLabelframe.Label", font=("Helvetica", 11, "bold"))
        style.configure("Log.TFrame", relief="sunken", borderwidth=1)
        style.configure("Control.TButton", font=("Helvetica", 11, "bold"), padding=8)

    def update_quantization_setting(self):
        """
        Update the configuration with the current quantization checkbox value.
        """
        self.config.perform_quantization = self.perform_quantization_var.get()

        # Keep the default script path
        if not hasattr(self.config, 'quantization_script_path') or not self.config.quantization_script_path:
            self.config.quantization_script_path = "../quantization/main.py"

        self.log_message(f"Quantization option {'enabled' if self.config.perform_quantization else 'disabled'}",
                         msg_type="info")

    def create_control_panel(self):
        """
        Create the training control panel with start/stop buttons and status indicators.
        """
        control_frame = ttk.LabelFrame(self.right_frame, text=" Training Controls ", padding="10")
        control_frame.pack(fill='x', padx=5, pady=5)  # Fill horizontally only

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill='x', padx=5, pady=5)  # Fill horizontally only

        self.start_button = ttk.Button(
            button_frame,
            text="▶ Start Training",
            command=self.start_training,
            style="Control.TButton",
            width=15
        )
        self.start_button.pack(side='left', padx=10)

        self.stop_button = ttk.Button(
            button_frame,
            text="■ Stop Training",
            command=self.stop_training,
            state='disabled',
            style="Control.TButton",
            width=15
        )
        self.stop_button.pack(side='left', padx=10)

        # Add the quantization checkbox above the status frame
        quant_frame = ttk.Frame(control_frame)
        quant_frame.pack(fill='x', padx=15, pady=5)  # Fill horizontally only

        # Create the BooleanVar and checkbox
        self.perform_quantization_var = tk.BooleanVar(value=self.config.perform_quantization)
        self.quantization_checkbutton = ttk.Checkbutton(
            quant_frame,
            text="Perform Quantization After Training",
            variable=self.perform_quantization_var,
            command=self.update_quantization_setting
        )
        self.quantization_checkbutton.pack(anchor=tk.W, padx=5, pady=5)

        # Add status frame below the quantization checkbox
        self.status_frame = ttk.Frame(control_frame)
        self.status_frame.pack(fill='x', padx=15, pady=5)  # Fill horizontally only

        status_row = ttk.Frame(self.status_frame)
        status_row.pack(fill='x')

        self.status_canvas = tk.Canvas(status_row, width=15, height=15)
        self.status_canvas.pack(side='left', padx=5)
        self.status_indicator = self.status_canvas.create_oval(2, 2, 13, 13, fill="green")

        self.status_label = ttk.Label(status_row, text="Status: Ready", font=("Helvetica", 10, "bold"))
        self.status_label.pack(side='left')

        # Initialize the quantization checkbox state from the config
        self.perform_quantization_var.set(self.config.perform_quantization)

        # Ensure the default script path is set
        if not hasattr(self.config, 'quantization_script_path') or not self.config.quantization_script_path:
            self.config.quantization_script_path = "../quantization/main.py"

    def create_log_area(self):
        """
        Create the log display area for training messages.
        """
        log_frame = ttk.LabelFrame(self.right_frame, text=" Training Log ", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrolled text widget for logs
        self.log_text = scrolledtext.ScrolledText(log_frame, height=18, width=25, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure tags for message types
        self.log_text.tag_configure("timestamp", foreground="blue")
        self.log_text.tag_configure("info", foreground="black")
        self.log_text.tag_configure("success", foreground="green")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("error", foreground="red")

        # Set background and border
        self.log_text.config(background="#f8f8f8", borderwidth=1, relief=tk.SOLID)
        self.log_message("System initialized and ready")

    def on_config_update(self, new_config):
        """
        Callback for configuration updates from the UI.
        """
        self.config = new_config

        # Update the quantization checkbox to match the config
        if hasattr(self, 'perform_quantization_var'):
            self.perform_quantization_var.set(self.config.perform_quantization)

        self.log_message("Configuration updated", msg_type="info")

    def log_message(self, message, msg_type="info"):
        """
        Add a timestamped message to the log area.

        Args:
            message (str): The message to log.
            msg_type (str): Type of message for coloring (info, success, warning, error).
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert('end', f"[{timestamp}] ", "timestamp")
        self.log_text.insert('end', f"{message}\n", msg_type)
        self.log_text.see('end')

    def start_training(self):
        """
        Initiate the training process in a separate thread.
        """
        if self.is_training:
            return

        self.is_training = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="Status: Training")
        self.status_canvas.itemconfig(self.status_indicator, fill="red")

        # Log training start and configuration details
        self.log_message("Starting training process...", msg_type="info")
        self.log_message("Training configuration:", msg_type="info")
        self.log_message(f"  - Input Length: {self.config.input_length}", msg_type="info")
        self.log_message(f"  - Filter Length: {self.config.filter_length}", msg_type="info")
        self.log_message(f"  - Network Layers: {self.config.num_filters_list}", msg_type="info")
        self.log_message(f"  - Max Epochs: {self.config.max_epochs}", msg_type="info")
        self.log_message(f"  - Batch Size: {self.config.batch_size}", msg_type="info")
        self.log_message(f"  - Cross Validation Folds: {self.config.num_folds}", msg_type="info")
        # if self.config.enable_regularization:
        #     self.log_message(f"  - L2 Regularization: Enabled (weight_decay_A={self.config.weight_decay_A}, (weight_decay_w={self.config.weight_decay_w}))",
        #                      msg_type="info")
        # else:
        #     self.log_message(f"  - L2 Regularization: Disabled", msg_type="info")

        # Start training in a separate thread to keep GUI responsive
        self.training_thread = threading.Thread(target=self.training_process)
        self.training_thread.start()
        self.root.after(100, self.process_messages)

    def stop_training(self):
        """
        Stop the training process.
        """
        if not self.is_training:
            return

        self.is_training = False
        self.status_label.config(text="Status: Stopping...")
        self.status_canvas.itemconfig(self.status_indicator, fill="orange")
        self.log_message("Stopping training. Please wait for current operations to complete...", msg_type="warning")

    def process_messages(self):
        """
        Process messages from the training thread and update the log.
        """
        while not self.message_queue.empty():
            message = self.message_queue.get()
            # Determine message type based on content
            if "error" in message.lower() or "failed" in message.lower():
                msg_type = "error"
            elif "warning" in message.lower() or "stopping" in message.lower():
                msg_type = "warning"
            elif "complete" in message.lower() or "success" in message.lower() or "accuracy" in message.lower():
                msg_type = "success"
            else:
                msg_type = "info"
            self.log_message(message, msg_type=msg_type)

        # Continue checking messages if training is active
        if self.is_training:
            self.root.after(100, self.process_messages)

    def load_custom_model(self, model_path, num_classes):
        """
        Load a custom model from a user-provided path.

        Args:
            model_path (str): Path to the custom model module.
            num_classes (int): Number of classes for the model.

        Returns:
            The loaded custom model.

        Raises:
            Exception: If the model cannot be loaded.
        """
        try:
            spec = importlib.util.spec_from_file_location("custom_model", model_path)
            custom_model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_model_module)
            model_class = getattr(custom_model_module, "CustomModel")
            model = model_class(num_classes=num_classes)
            return model
        except Exception as e:
            self.message_queue.put(f"Error loading custom model: {str(e)}")
            raise

    def training_process(self):
        """
        Main controller for the training process.
        Runs in a separate thread to manage training without freezing the GUI.
        """
        try:
            # Initialize training components
            data_manager, sigData, labelData, cv_indices, num_classes, trainer = self.initialize_training()

            # Perform cross-validation
            best_fold, best_accuracy, best_model_state, best_fold_data, accuracies, all_confusion_matrices = (
                self.run_cross_validation(sigData, labelData, cv_indices, num_classes, trainer)
            )

            # Process results if training hasn't been stopped
            if self.is_training:
                self.process_training_results(
                    best_fold, best_accuracy, best_model_state, best_fold_data,
                    accuracies, all_confusion_matrices, num_classes
                )

        except Exception as e:
            self.message_queue.put(f"Error during training process: {str(e)}")

        finally:
            self.is_training = False
            self.root.after(0, self.reset_buttons)

    def initialize_training(self):
        """
        Initialize training components and load data from JSON files.
        """
        # Set the data directory relative to the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_export_dir = os.path.join(parent_dir, "data")

        # Create a DataManager instance without relying on default_sig_path
        data_manager = DataManager(
            sig_path=None,  # No need for a .mat file path
            divide=self.config.num_folds,
            data_export_dir=data_export_dir
        )

        # Load and preprocess data from JSON files using custom paths if provided
        sigData, labelData, cv_indices, num_classes = data_manager.load_and_preprocess_data(
            sig_data_path=self.config.sig_data_path,
            label_data_path=self.config.label_data_path
        )

        if len(sigData.shape) == 3:
            # Shape is expected to be (num_samples, in_channels, data_length)
            self.config.in_channels = sigData.shape[1]
        elif len(sigData.shape) == 2:
            # Handle legacy 2D data by assuming 1 channel and reshaping
            self.message_queue.put("Warning: Loaded sigData is 2D. Assuming 1 input channel and reshaping.")
            sigData = np.expand_dims(sigData, axis=1)  # Add channel dimension
            self.config.in_channels = 1
        else:
            raise ValueError(f"Unexpected shape for sigData: {sigData.shape}")
        self.message_queue.put(f"Determined input channels: {self.config.in_channels}")

        output_dir = "../data"
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        in_channels_path = os.path.join(output_dir, "in_channels.json")

        try:
            with open(in_channels_path, 'w') as f:
                json.dump(self.config.in_channels, f)
            self.message_queue.put(f"Input channels ({self.config.in_channels}) exported to {in_channels_path}")
        except Exception as e:
            self.message_queue.put(f"Error exporting in_channels to JSON: {str(e)}")

        # # The following code exports cv_indices to JSON
        # output_dir = "../data"
        # os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
        # cv_indices_path = os.path.join(output_dir, "cv_indices.json")
        #
        # # Export cv_indices
        # with open(cv_indices_path, 'w') as f:
        #     json.dump(cv_indices.tolist(), f)

        # Export cv_indices to JSON (existing code)
        cv_indices_path = os.path.join(output_dir, "cv_indices.json")
        try:  # Added try-except for robustness
            with open(cv_indices_path, 'w') as f:
                json.dump(cv_indices.tolist(), f)
            self.message_queue.put(f"cv_indices exported to {cv_indices_path}")  # Optional log
        except Exception as e:
            self.message_queue.put(f"Error exporting cv_indices to JSON: {str(e)}")

        # Set up the training device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = ModelTrainer(self.config, device)

        # Log initialization details
        self.message_queue.put(f"Training with config: {self.config}")
        self.message_queue.put(f"Using device: {device}")
        self.message_queue.put(f"Signal data loaded from: {self.config.sig_data_path}")
        self.message_queue.put(f"Label data loaded from: {self.config.label_data_path}")

        return data_manager, sigData, labelData, cv_indices, num_classes, trainer

    def run_cross_validation(self, sigData, labelData, cv_indices, num_classes, trainer):
        """
        Run cross-validation training across multiple folds.

        Args:
            sigData: Signal data.
            labelData: Label data.
            cv_indices: Cross-validation indices.
            num_classes (int): Number of classes.
            trainer: ModelTrainer instance.

        Returns:
            Tuple containing best_fold, best_accuracy, best_model_state, best_fold_data, accuracies, all_confusion_matrices.
        """
        best_fold = 0
        best_accuracy = 0.0
        best_model_state = None
        best_fold_data = None
        accuracies = []
        all_confusion_matrices = []

        for iFold in range(1, self.config.num_folds + 1):
            if not self.is_training:
                self.message_queue.put("Training stopped")
                break

            self.message_queue.put(f"\n== Starting training for Fold {iFold} of {self.config.num_folds} ==")

            # Prepare data and train model
            dataloaders, current_fold_data = self.prepare_fold_data(sigData, labelData, cv_indices, iFold)
            train_loader, val_loader, test_loader = dataloaders
            model = self.initialize_model(num_classes)
            model.to(trainer.device)
            model_trained, test_accuracy, conf_matrix = trainer.train_model(model, train_loader, val_loader,
                                                                            test_loader)

            # Log and save fold results
            self.log_fold_results(iFold, conf_matrix, test_accuracy)
            current_fold_config = self.create_fold_config(iFold, test_accuracy, conf_matrix, num_classes)
            self.save_fold_data(iFold, model_trained, current_fold_config, current_fold_data)

            # Update best fold tracking
            if test_accuracy >= best_accuracy:
                best_accuracy = test_accuracy
                best_fold = iFold
                best_model_state = model_trained.state_dict()
                best_fold_data = current_fold_data

            accuracies.append(test_accuracy)
            all_confusion_matrices.append(conf_matrix)

        return best_fold, best_accuracy, best_model_state, best_fold_data, accuracies, all_confusion_matrices

    def prepare_fold_data(self, sigData, labelData, cv_indices, iFold):
        """
        Prepare data for a specific fold in cross-validation.

        Args:
            sigData: Signal data.
            labelData: Label data.
            cv_indices: Cross-validation indices.
            iFold (int): Current fold number.

        Returns:
            Tuple of (DataLoaders for train, val, test), current_fold_data dictionary.
        """
        # Split data into train/validation and test sets
        itest = (cv_indices == iFold)
        itrainval = ~itest
        XTrainVal = sigData[itrainval]
        YTrainVal = labelData[itrainval]
        itrain, ival = DataManager.get_train_val_idx(YTrainVal)
        XTrain = XTrainVal[itrain]
        YTrain = YTrainVal[itrain]
        XVal = XTrainVal[ival]
        YVal = YTrainVal[ival]
        XTest = sigData[itest]
        YTest = labelData[itest]

        # Convert to PyTorch tensors
        XTrain = torch.tensor(XTrain, dtype=torch.float32)
        YTrain = torch.tensor(YTrain, dtype=torch.long)
        XVal = torch.tensor(XVal, dtype=torch.float32)
        YVal = torch.tensor(YVal, dtype=torch.long)
        XTest = torch.tensor(XTest, dtype=torch.float32)
        YTest = torch.tensor(YTest, dtype=torch.long)

        def export_to_json(iFold, XTrain, YTrain, XVal, YVal, XTest, YTest):
            # Create dictionary with the data
            data_dict = {
                "iFold": int(iFold),
                "XTrain": XTrain.cpu().numpy().tolist(),
                "YTrain": YTrain.cpu().numpy().tolist(),
                "XVal": XVal.cpu().numpy().tolist(),
                "YVal": YVal.cpu().numpy().tolist(),
                "XTest": XTest.cpu().numpy().tolist(),
                "YTest": YTest.cpu().numpy().tolist()
            }

            # Create ../data directory if it doesn't exist
            os.makedirs("../data", exist_ok=True)

            # Define the output file path
            output_file = f"../data/fold_{iFold}_data.json"

            # Write to JSON file
            with open(output_file, 'w') as f:
                json.dump(data_dict, f)

            print(f"Data successfully exported to {output_file}")

        export_to_json(
            iFold,
            XTrain,
            YTrain,
            XVal,
            YVal,
            XTest,
            YTest
        )


        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(XTrain, YTrain), batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(XVal, YVal), batch_size=self.config.batch_size)
        test_loader = DataLoader(TensorDataset(XTest, YTest), batch_size=self.config.batch_size)

        # Store fold data
        current_fold_data = {
            'train_data': XTrain.cpu().numpy(),
            'train_labels': YTrain.cpu().numpy(),
            'val_data': XVal.cpu().numpy(),
            'val_labels': YVal.cpu().numpy(),
            'test_data': XTest.cpu().numpy(),
            'test_labels': YTest.cpu().numpy()
        }

        return (train_loader, val_loader, test_loader), current_fold_data

    def initialize_model(self, num_classes):
        """
        Initialize the model based on configuration.

        Args:
            num_classes (int): Number of classes for the model.

        Returns:
            Initialized model instance.
        """
        if self.config.model_type == "default":
            model = CosCNN(
                input_length=self.config.input_length,
                in_channels=self.config.in_channels,
                num_classes=num_classes,
                filter_length=self.config.filter_length,
                num_filters_list=self.config.num_filters_list
            )
            self.message_queue.put("Using default CosCNN model")
        else:
            model = self.load_custom_model(self.config.custom_model_path, num_classes)
            self.message_queue.put(f"Loaded custom model from {self.config.custom_model_path}")
        return model

    def log_fold_results(self, iFold, conf_matrix, test_accuracy):
        """
        Log results for a specific fold.

        Args:
            iFold (int): Fold number.
            conf_matrix: Confusion matrix.
            test_accuracy (float): Test accuracy.
        """
        self.message_queue.put(f"\nFold {iFold} Confusion Matrix:")
        conf_matrix_str = np.array2string(conf_matrix, precision=2, suppress_small=True)
        self.message_queue.put("\n" + conf_matrix_str)

        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        self.message_queue.put("Accuracy by Class:")
        for i, acc in enumerate(class_accuracies):
            self.message_queue.put(f"Class {i}: {acc * 100:.2f}%")

        self.message_queue.put(f'Fold {iFold}, Test Accuracy: {test_accuracy * 100:.2f}%')

    def create_fold_config(self, iFold, test_accuracy, conf_matrix, num_classes):
        """
        Create configuration dictionary for a specific fold.

        Args:
            iFold (int): Fold number.
            test_accuracy (float): Test accuracy.
            conf_matrix: Confusion matrix.
            num_classes (int): Number of classes.

        Returns:
            dict: Configuration for the fold.
        """
        return {
            'input_length': self.config.input_length,
            'num_classes': num_classes,
            'filter_length': self.config.filter_length,
            'num_filters_list': self.config.num_filters_list,
            'fold_number': iFold,
            'accuracy': test_accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'date_saved': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def save_fold_data(self, iFold, model, fold_config, fold_data):
        """
        Save data for a specific fold.

        Args:
            iFold (int): Fold number.
            model: Trained model.
            fold_config (dict): Fold configuration.
            fold_data (dict): Fold data dictionary.
        """
        # parent_dir = os.path.dirname(self.default_sig_path)
        parent_dir = os.getcwd()  # Get current working directory
        save_dir = os.path.join(parent_dir, '../trained_models')
        os.makedirs(save_dir, exist_ok=True)

        fold_save_path = os.path.join(save_dir, f'fold_{iFold}')
        os.makedirs(fold_save_path, exist_ok=True)

        # Save model state and configuration
        model_state = {'state_dict': model.state_dict(), 'config': fold_config}
        torch.save(model_state, os.path.join(fold_save_path, f'model_fold_{iFold}.pth'))
        with open(os.path.join(fold_save_path, 'config.json'), 'w') as f:
            json.dump(fold_config, f, indent=4)

        # Save fold data
        np.savez(
            os.path.join(fold_save_path, 'data.npz'),
            train_data=fold_data['train_data'],
            train_labels=fold_data['train_labels'],
            val_data=fold_data['val_data'],
            val_labels=fold_data['val_labels'],
            test_data=fold_data['test_data'],
            test_labels=fold_data['test_labels']
        )

    def process_training_results(self, best_fold, best_accuracy, best_model_state, best_fold_data,
                                 accuracies, all_confusion_matrices, num_classes):
        """
        Process and save final training results.

        Args:
            best_fold (int): Best performing fold number.
            best_accuracy (float): Best accuracy.
            best_model_state: State of the best model.
            best_fold_data (dict): Data of the best fold.
            accuracies (list): List of accuracies from all folds.
            all_confusion_matrices (list): List of confusion matrices from all folds.
            num_classes (int): Number of classes.
        """
        self.log_cross_validation_results(accuracies, all_confusion_matrices)
        self.save_best_model(best_fold, best_accuracy, best_model_state, best_fold_data,
                             accuracies, all_confusion_matrices, num_classes)

        # Run quantization if specified
        if self.config.perform_quantization:
            self.run_quantization()

    def log_cross_validation_results(self, accuracies, all_confusion_matrices):
        """
        Log overall cross-validation results.

        Args:
            accuracies (list): List of accuracies from all folds.
            all_confusion_matrices (list): List of confusion matrices from all folds.
        """
        self.message_queue.put("═" * 50)
        self.message_queue.put(f"             {self.config.num_folds}-Fold Cross-Validation Results")
        self.message_queue.put("═" * 50)
        self.message_queue.put(f"{'Fold':^15} | {'Accuracy (%)':^15}")
        self.message_queue.put("─" * 50)

        for i, acc in enumerate(accuracies, 1):
            self.message_queue.put(f"{'Fold ' + str(i):^15} | {acc * 100:^15.2f}")

        self.message_queue.put("─" * 50)
        self.message_queue.put(f"{'Mean Accuracy':^15} | {np.mean(accuracies) * 100:^15.2f}")
        self.message_queue.put(f"{'Std':^15} | {np.std(accuracies) * 100:^15.2f}%")
        self.message_queue.put("═" * 50)

        avg_conf_matrix = np.mean(all_confusion_matrices, axis=0)
        self.message_queue.put("\nAverage Confusion Matrix (All Folds):")
        avg_conf_matrix_str = np.array2string(avg_conf_matrix, precision=2, suppress_small=True)
        self.message_queue.put("\n" + avg_conf_matrix_str)

        self.message_queue.put("\nCross-Validation Accuracy Statistics:")
        self.message_queue.put(f"Mean Accuracy: {np.mean(accuracies) * 100:.2f}%")
        self.message_queue.put(f"Standard Deviation: {np.std(accuracies) * 100:.2f}%")
        self.message_queue.put(f"Maximum Accuracy: {np.max(accuracies) * 100:.2f}%")
        self.message_queue.put(f"Minimum Accuracy: {np.min(accuracies) * 100:.2f}%")

    def save_best_model(self, best_fold, best_accuracy, best_model_state, best_fold_data,
                        accuracies, all_confusion_matrices, num_classes):
        """
        Save the best performing model from all folds.

        Args:
            best_fold (int): Best performing fold number.
            best_accuracy (float): Best accuracy.
            best_model_state: State of the best model.
            best_fold_data (dict): Data of the best fold.
            accuracies (list): List of accuracies from all folds.
            all_confusion_matrices (list): List of confusion matrices from all folds.
            num_classes (int): Number of classes.
        """
        avg_conf_matrix = np.mean(all_confusion_matrices, axis=0)
        model_config = {
            'input_length': self.config.input_length,
            'num_classes': num_classes,
            'filter_length': self.config.filter_length,
            'num_filters_list': self.config.num_filters_list,
            'best_fold': best_fold,
            'best_accuracy': best_accuracy,
            'average_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'average_confusion_matrix': avg_conf_matrix.tolist(),
            'date_saved': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # parent_dir = os.path.dirname(self.default_sig_path)
        parent_dir = os.getcwd()  # Get current working directory
        save_dir = os.path.join(parent_dir, '../trained_models')
        best_model_path = os.path.join(save_dir, 'best_model')
        os.makedirs(best_model_path, exist_ok=True)

        # Save model state and configuration
        torch.save(best_model_state, os.path.join(best_model_path, 'model.pth'))
        with open(os.path.join(best_model_path, 'config.json'), 'w') as f:
            json.dump(model_config, f, indent=4)

        # Save best fold data
        np.savez(
            os.path.join(best_model_path, 'data.npz'),
            train_data=best_fold_data['train_data'],
            train_labels=best_fold_data['train_labels'],
            val_data=best_fold_data['val_data'],
            val_labels=best_fold_data['val_labels'],
            test_data=best_fold_data['test_data'],
            test_labels=best_fold_data['test_labels']
        )

        # Log completion and best model details
        self.message_queue.put("\n✅ Training complete!")
        self.message_queue.put(f"\nBest Model Information:")
        self.message_queue.put(f"Best Performing Fold: {best_fold}")
        self.message_queue.put(f"Best Test Accuracy: {best_accuracy * 100:.2f}%")
        self.message_queue.put(f"Average Test Accuracy: {np.mean(accuracies) * 100:.2f}%")

    def run_quantization(self):
        """
        Execute the quantization process using an external script.
        """
        self.message_queue.put("\n⚙️ Starting Quantization...")
        self.status_label.config(text="Status: Quantizing")
        quantization_script = self.config.quantization_script_path
        try:
            process = subprocess.Popen(
                ["python", quantization_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                errors='replace'
            )

            def read_output(pipe, queue):
                """Read output from subprocess and queue messages."""
                for line in iter(pipe.readline, ''):
                    if line:
                        import re
                        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                        clean_line = ansi_escape.sub('', line)
                        clean_line = re.sub(r'[\x00-\x1F\x7F]', ' ', clean_line).strip()
                        if clean_line:
                            queue.put(clean_line)

            # Threads to capture stdout and stderr
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, self.message_queue))
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, self.message_queue))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            process.wait()
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)

            # Log quantization result
            if process.returncode == 0:
                self.message_queue.put("✅ Quantization Finished")
            else:
                self.message_queue.put(f"❌ Quantization Failed with return code: {process.returncode}")

        except Exception as e:
            self.message_queue.put(f"Cannot run quantization script: {str(e)}")

    def reset_buttons(self):
        """
        Reset the state of control buttons and status indicator.
        """
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Status: Ready")
        self.status_canvas.itemconfig(self.status_indicator, fill="green")

    def run(self):
        """
        Run the application by centering the window and starting the main loop.
        """
        self.center_window()
        self.root.mainloop()

    def center_window(self):
        """
        Center the application window on the screen.
        """
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))


def main():
    """
    Main entry point for the application.
    Creates and runs the TrainingApplication instance.
    """
    app = TrainingApplication()
    app.run()


if __name__ == '__main__':
    main()