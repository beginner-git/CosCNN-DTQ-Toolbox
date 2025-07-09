import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from config import TrainingConfig
from typing import Callable, Optional
import json
import os

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            tw,
            text=self.text,
            background="#f0f8ff",
            foreground="#333333",
            relief="solid",
            borderwidth=1,
            font=("Helvetica", 9),
            padding=(6, 3)
        )
        label.pack()

    def hide_tip(self, event):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class TrainingConfigUI:
    def __init__(self, parent, on_config_update: Optional[Callable[[TrainingConfig], None]] = None):
        """Initialize the training configuration UI"""
        self.parent = parent
        self.on_config_update = on_config_update
        self.config = TrainingConfig()

        # Configure styles
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'vista' in available_themes:
            style.theme_use('vista')
        elif 'clearlooks' in available_themes:
            style.theme_use('clearlooks')
        elif 'aqua' in available_themes:
            style.theme_use('aqua')
        else:
            style.theme_use('clam')

        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("TLabelframe", font=("Helvetica", 11, "bold"))
        style.configure("TLabelframe.Label", font=("Helvetica", 11, "bold"))
        style.configure("TRadiobutton", font=("Helvetica", 10))
        style.configure("TCheckbutton", font=("Helvetica", 10))
        style.configure("TEntry", padding=3)

        # Create main frame
        self.main_frame = ttk.Frame(parent, padding="0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=0)
        self.main_frame.columnconfigure(0, weight=1)

        # Create notebook widget
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Create tabs (only two tabs now)
        self.network_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.network_tab, text="Model Settings")
        self.notebook.add(self.training_tab, text="Training Parameters")

        # Initialize tab content
        self.create_network_frame(self.network_tab)
        self.create_training_frame(self.training_tab)
        self.create_buttons_frame()

        self.ensure_editable()

    def create_network_frame(self, parent):
        """Create network structure configuration frame"""
        parent.columnconfigure(1, weight=1)

        # Model selection
        self.model_type_var = tk.StringVar(value="default")
        radio_frame = ttk.Frame(parent)
        radio_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        default_radio = ttk.Radiobutton(radio_frame, text="Design Model", variable=self.model_type_var,
                                        value="default")
        default_radio.pack(side=tk.LEFT, padx=5)
        ToolTip(default_radio, "Use the default CosCNN model with configurable layers")
        custom_radio = ttk.Radiobutton(radio_frame, text="Load Model", variable=self.model_type_var, value="custom")
        custom_radio.pack(side=tk.LEFT, padx=5)
        ToolTip(custom_radio, "Use a custom model defined in a Python file")

        # Create a label frame for data settings
        data_label_frame = ttk.LabelFrame(parent, text="Data Settings", padding=(10, 5, 10, 10))
        data_label_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        data_label_frame.columnconfigure(1, weight=1)  # Make the entry column expandable

        # Signal data path
        ttk.Label(data_label_frame, text="Signal Data Path:", width=15).grid(row=0, column=0, sticky=tk.W, padx=5,
                                                                             pady=3)
        self.sig_data_path_var = tk.StringVar(value="../data/sigData.json")
        self.sig_data_entry = ttk.Entry(data_label_frame, textvariable=self.sig_data_path_var, width=20)
        self.sig_data_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=3)
        self.browse_sig_data_btn = ttk.Button(data_label_frame, text="Browse", command=self.browse_sig_data_file,
                                              width=10)
        self.browse_sig_data_btn.grid(row=0, column=2, padx=5, pady=3)
        ToolTip(self.sig_data_entry, "Path to signal data JSON file")

        # Label data path
        ttk.Label(data_label_frame, text="Label Data Path:", width=15).grid(row=1, column=0, sticky=tk.W, padx=5,
                                                                            pady=3)
        self.label_data_path_var = tk.StringVar(value="../data/labelData.json")
        self.label_data_entry = ttk.Entry(data_label_frame, textvariable=self.label_data_path_var, width=20)
        self.label_data_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=3)
        self.browse_label_data_btn = ttk.Button(data_label_frame, text="Browse", command=self.browse_label_data_file,
                                                width=10)
        self.browse_label_data_btn.grid(row=1, column=2, padx=5, pady=3)
        ToolTip(self.label_data_entry, "Path to label data JSON file")

        # Input length (read-only) with Load Data button next to it
        ttk.Label(data_label_frame, text="Input Length:", width=15).grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.input_length_var = tk.StringVar(value=str(self.config.input_length))
        input_entry = ttk.Entry(data_label_frame, textvariable=self.input_length_var, width=10, state='readonly')
        input_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)
        ToolTip(input_entry, "Length of input data, calculated from loaded signal data")

        # Add load button in the same row as Input Length
        self.load_data_btn = ttk.Button(data_label_frame, text="Load Data", command=self.load_signal_data, width=10)
        self.load_data_btn.grid(row=2, column=2, padx=5, pady=3)
        ToolTip(self.load_data_btn, "Load signal data and calculate input length")

        # Custom model path in a separate section
        model_label_frame = ttk.LabelFrame(parent, text="Model Settings", padding=(10, 5, 10, 10))
        model_label_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        model_label_frame.columnconfigure(1, weight=1)  # Make the entry column expandable

        ttk.Label(model_label_frame, text="Model Path:", width=15).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.custom_model_path_var = tk.StringVar()
        self.custom_path_entry = ttk.Entry(model_label_frame, textvariable=self.custom_model_path_var, width=20,
                                           state='disabled')
        self.custom_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=3)
        self.browse_model_btn = ttk.Button(model_label_frame, text="Browse", command=self.browse_model_file, width=10,
                                           state='disabled')
        self.browse_model_btn.grid(row=0, column=2, padx=5, pady=3)

        # Filter length
        ttk.Label(model_label_frame, text="Kernel Length:", width=15).grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.filter_length_var = tk.StringVar(value=str(self.config.filter_length))
        self.filter_entry = ttk.Entry(model_label_frame, textvariable=self.filter_length_var, width=10)
        self.filter_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
        ToolTip(self.filter_entry, "Length of convolution filter, must be a positive integer")

        # Network layers configuration - IMPROVED LAYOUT
        layer_config_frame = ttk.LabelFrame(parent, text="Model Designer", padding=(5, 5, 5, 5))
        layer_config_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=(10, 5))
        layer_config_frame.columnconfigure(0, weight=1)

        # Container for layers with add/remove buttons to the right
        container_frame = ttk.Frame(layer_config_frame)
        container_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        container_frame.columnconfigure(0, weight=1)

        # Layers frame for the layer entries (left side)
        self.layers_frame = ttk.Frame(container_frame)
        self.layers_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.layers_frame.columnconfigure(1, weight=1)

        # Button frame for add/remove buttons (right side)
        btn_frame = ttk.Frame(container_frame)
        btn_frame.grid(row=0, column=1, sticky=(tk.N, tk.E), padx=(5, 0))

        # Stack buttons vertically with descriptive text
        self.add_layer_btn = ttk.Button(btn_frame, text="Add Layer", command=self.add_layer, width=12)
        self.add_layer_btn.grid(row=0, column=0, pady=(0, 3))
        ToolTip(self.add_layer_btn, "Add a new layer")

        self.remove_layer_btn = ttk.Button(btn_frame, text="Remove Layer", command=self.remove_layer, width=12)
        self.remove_layer_btn.grid(row=1, column=0)
        ToolTip(self.remove_layer_btn, "Remove the last layer")

        self.layer_vars = []
        self.layer_entries = []
        self.update_layers_ui()

        # Set initial state and add trace
        self.model_type_var.trace("w", lambda *args: self.toggle_network_config_widgets())
        self.toggle_network_config_widgets()

    def load_signal_data(self):
        """Load signal data to calculate input length from the third dimension"""
        try:
            sig_json_path = self.sig_data_path_var.get()

            # Check if file exists
            if not os.path.exists(sig_json_path):
                messagebox.showerror("Error", f"Signal data file not found: {sig_json_path}", icon="error")
                return

            # Load signal data file
            with open(sig_json_path, 'r') as f:
                sig_data = json.load(f)

            input_length = 0  # Initialize with a default value
            if (isinstance(sig_data, list) and len(sig_data) > 0 and  # Check outer list
                    isinstance(sig_data[0], list) and len(sig_data[0]) > 0 and  # Check middle list
                    isinstance(sig_data[0][0], list) and len(sig_data[0][0]) > 0):  # Check inner list
                input_length = len(sig_data[0][0])  # Get length of the innermost list

            # Check if input_length was successfully calculated
            if input_length > 0:
                # Update input length variable and config
                self.input_length_var.set(str(input_length))
                self.config.input_length = input_length  # Assuming self.config exists

                messagebox.showinfo("Success", f"Input length calculated: {input_length}", icon="info")
            else:
                # More specific error message if structure is wrong or lists are empty
                messagebox.showerror("Error",
                                     "Invalid signal data format or empty data. Expected 3D list [num_samples, 1, length].",
                                     icon="error")

        except json.JSONDecodeError:
            messagebox.showerror("Error", f"Invalid JSON format in {sig_json_path}", icon="error")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process signal data: {str(e)}", icon="error")

    def browse_sig_data_file(self):
        """Browse and select signal data file"""
        file_path = filedialog.askopenfilename(
            title="Select Signal Data File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.sig_data_path_var.set(file_path)

    def browse_label_data_file(self):
        """Browse and select label data file"""
        file_path = filedialog.askopenfilename(
            title="Select Label Data File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.label_data_path_var.set(file_path)

    def update_layers_ui(self):
        """Update network layer configuration UI"""
        for widget in self.layers_frame.winfo_children():
            widget.destroy()
        self.layer_vars.clear()
        self.layer_entries.clear()
        model_type = self.model_type_var.get()
        state = 'normal' if model_type == "default" else 'disabled'

        # Create a header row
        header = ttk.Label(self.layers_frame, text="Layer Configuration", font=("Helvetica", 10, "bold"))
        header.grid(row=0, column=0, columnspan=2, sticky=(tk.W), pady=(0, 5))

        for i, filters in enumerate(self.config.num_filters_list):
            row = i + 1  # Start from row 1 after the header
            ttk.Label(self.layers_frame, text=f"Layer {i + 1} Channels:", width=16).grid(
                row=row, column=0, sticky=tk.W, padx=5, pady=2
            )
            var = tk.StringVar(value=str(filters))
            self.layer_vars.append(var)
            entry = ttk.Entry(self.layers_frame, textvariable=var, width=10, state=state)
            entry.grid(row=row, column=1, sticky=(tk.W), padx=5, pady=2)
            self.layer_entries.append(entry)
            ToolTip(entry, f"Number of output channels for layer {i + 1}")

    def toggle_network_config_widgets(self):
        """Toggle network configuration widget states based on model type"""
        model_type = self.model_type_var.get()
        if model_type == "default":
            state = 'normal'
            path_state = 'disabled'
        else:
            state = 'disabled'
            path_state = 'normal'
        self.filter_entry.config(state=state)
        for entry in self.layer_entries:
            entry.config(state=state)
        self.add_layer_btn.config(state=state)
        self.remove_layer_btn.config(state=state)
        self.custom_path_entry.config(state=path_state)
        self.browse_model_btn.config(state=path_state)

    def add_layer(self):
        """Add a network layer"""
        self.config.num_filters_list.append(64)
        self.update_layers_ui()

    def remove_layer(self):
        """Remove a network layer"""
        if len(self.config.num_filters_list) > 1:
            self.config.num_filters_list.pop()
            self.update_layers_ui()
        else:
            messagebox.showwarning("Warning", "At least one network layer must be retained!", icon="warning")

    def create_training_frame(self, parent):
        """Create training parameters configuration frame"""
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Max Training Epochs:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_epochs_var = tk.StringVar(value=str(self.config.max_epochs))
        epochs_entry = ttk.Entry(parent, textvariable=self.max_epochs_var, width=15)
        epochs_entry.grid(row=0, column=1, sticky=(tk.W), padx=5, pady=5)
        ToolTip(epochs_entry, "Maximum number of training epochs")
        ttk.Label(parent, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_size_var = tk.StringVar(value=str(self.config.batch_size))
        batch_entry = ttk.Entry(parent, textvariable=self.batch_size_var, width=15)
        batch_entry.grid(row=1, column=1, sticky=(tk.W), padx=5, pady=5)
        ToolTip(batch_entry, "Number of samples per training batch")
        ttk.Label(parent, text="Initial Learning Rate:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.lr_start_var = tk.StringVar(value=str(self.config.lr_start))
        lr_start_entry = ttk.Entry(parent, textvariable=self.lr_start_var, width=15)
        lr_start_entry.grid(row=2, column=1, sticky=(tk.W), padx=5, pady=5)
        ToolTip(lr_start_entry, "Learning rate at the start of training")
        ttk.Label(parent, text="Final Learning Rate:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.lr_end_var = tk.StringVar(value=str(self.config.lr_end))
        lr_end_entry = ttk.Entry(parent, textvariable=self.lr_end_var, width=15)
        lr_end_entry.grid(row=3, column=1, sticky=(tk.W), padx=5, pady=5)
        ToolTip(lr_end_entry, "Learning rate at the end of training")
        ttk.Label(parent, text="LR Drop Period:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.lr_drop_period_var = tk.StringVar(value=str(self.config.lr_drop_period))
        lr_drop_entry = ttk.Entry(parent, textvariable=self.lr_drop_period_var, width=15)
        lr_drop_entry.grid(row=4, column=1, sticky=(tk.W), padx=5, pady=5)
        ToolTip(lr_drop_entry, "Number of epochs between learning rate drops")
        ttk.Label(parent, text="Number of Folds:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.num_folds_var = tk.StringVar(value=str(self.config.num_folds))
        folds_entry = ttk.Entry(parent, textvariable=self.num_folds_var, width=15)
        folds_entry.grid(row=5, column=1, sticky=(tk.W), padx=5, pady=5)
        ToolTip(folds_entry, "Number of folds for cross-validation")

        # regularization_frame = ttk.LabelFrame(parent, text="Regularization", padding="10")
        # regularization_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=10)

        # Create frame for learnable parameter settings
        regularization_frame = ttk.LabelFrame(parent, text="Learnable Parameter Settings", padding="10")
        regularization_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=10)

        # # Add checkbox
        # self.enable_regularization_var = tk.BooleanVar(value=self.config.enable_regularization)
        # enable_reg_check = ttk.Checkbutton(
        #     regularization_frame,
        #     text="Enable L2 Regularization",
        #     variable=self.enable_regularization_var,
        #     command=self.toggle_regularization_fields  # Bind callback function
        # )
        # enable_reg_check.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        # Parameter 1: Weight Decay (A)
        ttk.Label(regularization_frame, text="Regularization Coefficient (A):").grid(row=2, column=0, sticky=tk.W,
                                                                                     padx=5, pady=5)
        self.weight_decay_A_var = tk.StringVar(value=str(self.config.weight_decay_A))
        self.weight_decay_A_entry = ttk.Entry(regularization_frame, textvariable=self.weight_decay_A_var, width=15)
        self.weight_decay_A_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Parameter 2: Weight Decay (w)
        ttk.Label(regularization_frame, text="Regularization Coefficient (w):").grid(row=3, column=0, sticky=tk.W,
                                                                                     padx=5, pady=5)
        self.weight_decay_w_var = tk.StringVar(value=str(self.config.weight_decay_w))
        self.weight_decay_w_entry = ttk.Entry(regularization_frame, textvariable=self.weight_decay_w_var, width=15)
        self.weight_decay_w_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

        # Parameter 3: Learning Rate Factor (A)
        ttk.Label(regularization_frame, text="Learning Rate Factor (A):").grid(row=4, column=0, sticky=tk.W, padx=5,
                                                                               pady=5)
        self.lr_factor_A_var = tk.StringVar(value=str(self.config.lr_factor_A))
        self.lr_factor_A_entry = ttk.Entry(regularization_frame, textvariable=self.lr_factor_A_var, width=15)
        self.lr_factor_A_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)

        # Parameter 4: Learning Rate Factor (w)
        ttk.Label(regularization_frame, text="Learning Rate Factor (w):").grid(row=5, column=0, sticky=tk.W, padx=5,
                                                                               pady=5)
        self.lr_factor_w_var = tk.StringVar(value=str(self.config.lr_factor_w))
        self.lr_factor_w_entry = ttk.Entry(regularization_frame, textvariable=self.lr_factor_w_var, width=15)
        self.lr_factor_w_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)

        # # Set initial state
        # self.toggle_regularization_fields()

    def create_quantization_frame(self, parent):
        """Create quantization configuration frame"""
        parent.columnconfigure(1, weight=1)
        check_frame = ttk.Frame(parent, padding=5)
        check_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W), pady=5)
        self.perform_quantization_var = tk.BooleanVar(value=self.config.perform_quantization)
        self.quantization_checkbutton = ttk.Checkbutton(
            check_frame,
            text="Perform Quantization After Training",
            variable=self.perform_quantization_var
        )
        self.quantization_checkbutton.pack(anchor=tk.W, pady=5)
        ToolTip(self.quantization_checkbutton,
                "Enable to run model quantization automatically after training completes")
        path_frame = ttk.Frame(parent)
        path_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W), pady=10)
        ttk.Label(path_frame, text="Script Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.quantization_script_path_var = tk.StringVar(value=self.config.quantization_script_path)
        script_entry = ttk.Entry(path_frame, textvariable=self.quantization_script_path_var)
        script_entry.grid(row=0, column=1, sticky=(tk.W), padx=5, pady=5)
        ToolTip(script_entry, "Path to the quantization script Python file")
        browse_btn = ttk.Button(path_frame, text="Browse", command=self.browse_quantization_script, width=10)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        info_label = ttk.Label(
            parent,
            text="Quantization reduces model size and improves inference speed by converting floating-point weights to lower precision representations.",
            justify=tk.LEFT,
            wraplength=350,
            font=("Helvetica", 9)
        )
        info_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=15)

    def create_buttons_frame(self):
        """Create action buttons frame"""
        buttons_frame = ttk.Frame(self.main_frame, padding="10")
        buttons_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        buttons_frame.columnconfigure((0, 1, 2), weight=1)
        save_btn = ttk.Button(buttons_frame, text="💾 Save Config.", command=self.save_config, width=15)
        save_btn.grid(row=0, column=0, padx=0)
        load_btn = ttk.Button(buttons_frame, text="📂 Load Config.", command=self.load_config, width=15)
        load_btn.grid(row=0, column=1, padx=0)
        apply_btn = ttk.Button(buttons_frame, text="✓ Apply Config.", command=self.apply_config, width=15)
        apply_btn.grid(row=0, column=2, padx=0)

    def save_config(self):
        """Save configuration to file"""
        try:
            config_dict = {
                'input_length': int(self.input_length_var.get()),
                'filter_length': int(self.filter_length_var.get()),
                'num_filters_list': [int(var.get()) for var in self.layer_vars],
                'max_epochs': int(self.max_epochs_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'lr_start': float(self.lr_start_var.get()),
                'lr_end': float(self.lr_end_var.get()),
                'lr_drop_period': int(self.lr_drop_period_var.get()),
                'num_folds': int(self.num_folds_var.get()),
                'model_type': self.model_type_var.get(),
                'custom_model_path': self.custom_model_path_var.get() if self.model_type_var.get() == "custom" else None,
                # Keep the quantization settings without UI elements
                'perform_quantization': self.config.perform_quantization,
                'quantization_script_path': "../quantization/main.py",  # Use default path
                'weight_decay_A': float(self.weight_decay_A_var.get()),
                'weight_decay_w': float(self.weight_decay_w_var.get()),
                'lr_factor_A': float(self.lr_factor_A_var.get()),
                'lr_factor_w': float(self.lr_factor_w_var.get()),
                # Add data paths
                'sig_data_path': self.sig_data_path_var.get(),
                'label_data_path': self.label_data_path_var.get(),
            }
            if config_dict['input_length'] <= 0 or config_dict['filter_length'] <= 0:
                raise ValueError("Input length and filter length must be positive integers")
            if not all(f > 0 for f in config_dict['num_filters_list']):
                raise ValueError("All layer output channels must be positive integers")
            with open('training_config.json', 'w') as f:
                json.dump(config_dict, f, indent=4)
            messagebox.showinfo("Success", "Configuration saved to training_config.json", icon="info")
        except ValueError as ve:
            messagebox.showerror("Error", f"Invalid input: {str(ve)}", icon="error")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}", icon="error")

    def load_config(self):
        """Load configuration from file"""
        if os.path.exists('training_config.json'):
            try:
                with open('training_config.json', 'r') as f:
                    config_dict = json.load(f)

                # Store the quantization settings before updating the config
                perform_quantization = config_dict.get('perform_quantization', False)
                quant_script_path = config_dict.get('quantization_script_path', "../quantization/main.py")

                self.config.update(**config_dict)
                # Explicitly set the quantization settings
                self.config.perform_quantization = perform_quantization
                self.config.quantization_script_path = quant_script_path
                self.input_length_var.set(str(self.config.input_length))
                self.filter_length_var.set(str(self.config.filter_length))
                self.max_epochs_var.set(str(self.config.max_epochs))
                self.batch_size_var.set(str(self.config.batch_size))
                self.lr_start_var.set(str(self.config.lr_start))
                self.lr_end_var.set(str(self.config.lr_end))
                self.lr_drop_period_var.set(str(self.config.lr_drop_period))
                self.num_folds_var.set(str(self.config.num_folds))
                self.model_type_var.set(self.config.model_type)
                self.custom_model_path_var.set(self.config.custom_model_path or "")
                # Set data paths
                if 'sig_data_path' in config_dict:
                    self.sig_data_path_var.set(config_dict['sig_data_path'])
                if 'label_data_path' in config_dict:
                    self.label_data_path_var.set(config_dict['label_data_path'])
                # Remove this line: self.perform_quantization_var.set(self.config.perform_quantization)
                self.weight_decay_A_var.set(str(self.config.weight_decay_A))
                self.weight_decay_w_var.set(str(self.config.weight_decay_w))
                self.lr_factor_A_var.set(str(self.config.lr_factor_A))
                self.lr_factor_w_var.set(str(self.config.lr_factor_w))
                self.update_layers_ui()
                self.toggle_network_config_widgets()

                if self.on_config_update:
                    self.on_config_update(self.config)

                messagebox.showinfo("Success", "Configuration loaded successfully", icon="info")
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid configuration file format", icon="error")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}", icon="error")
        else:
            messagebox.showinfo("Information", "Configuration file not found, using default values", icon="info")

    def apply_config(self):
        """Apply current configuration"""
        try:
            config_dict = {
                'input_length': int(self.input_length_var.get()),
                'filter_length': int(self.filter_length_var.get()),
                'num_filters_list': [int(var.get()) for var in self.layer_vars],
                'max_epochs': int(self.max_epochs_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'lr_start': float(self.lr_start_var.get()),
                'lr_end': float(self.lr_end_var.get()),
                'lr_drop_period': int(self.lr_drop_period_var.get()),
                'num_folds': int(self.num_folds_var.get()),
                'model_type': self.model_type_var.get(),
                'custom_model_path': self.custom_model_path_var.get() if self.model_type_var.get() == "custom" else None,
                # Keep the quantization settings without UI elements
                'perform_quantization': self.config.perform_quantization,
                'quantization_script_path': "../quantization/main.py",
                'weight_decay_A': float(self.weight_decay_A_var.get()),
                'weight_decay_w': float(self.weight_decay_w_var.get()),
                'lr_factor_A': float(self.lr_factor_A_var.get()),
                'lr_factor_w': float(self.lr_factor_w_var.get()),
                # Add data paths
                'sig_data_path': self.sig_data_path_var.get(),
                'label_data_path': self.label_data_path_var.get(),
            }
            if config_dict['input_length'] <= 0 or config_dict['filter_length'] <= 0:
                raise ValueError("Input length and filter length must be positive integers")
            if not all(f > 0 for f in config_dict['num_filters_list']):
                raise ValueError("All layer output channels must be positive integers")
            self.config.update(**config_dict)
            if self.on_config_update:
                self.on_config_update(self.config)
            messagebox.showinfo("Success", "Configuration applied successfully", icon="info")
        except ValueError as ve:
            messagebox.showerror("Error", f"Invalid input: {str(ve)}", icon="error")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply configuration: {str(e)}", icon="error")

    def browse_model_file(self):
        """Browse and select custom model file"""
        file_path = filedialog.askopenfilename(
            title="Select Custom Model File",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if file_path:
            self.custom_model_path_var.set(file_path)

    def browse_quantization_script(self):
        """Browse and select quantization script file"""
        file_path = filedialog.askopenfilename(
            title="Select Quantization Script",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if file_path:
            self.quantization_script_path_var.set(file_path)

    def ensure_editable(self):
        """Ensure all input widgets are editable"""
        for frame in [self.main_frame]:
            for widget in frame.winfo_children():
                if isinstance(widget, (ttk.Entry, ttk.Checkbutton, ttk.Radiobutton)):
                    widget.config(state='normal')

    # def toggle_regularization_fields(self):
    #     """Enable or disable regularization parameter input boxes based on checkbox state"""
    #     state = 'normal' if self.enable_regularization_var.get() else 'disabled'
    #     self.weight_decay_A_entry.config(state=state)
    #     self.weight_decay_w_entry.config(state=state)
    #     self.lr_factor_A_entry.config(state=state)
    #     self.lr_factor_w_entry.config(state=state)