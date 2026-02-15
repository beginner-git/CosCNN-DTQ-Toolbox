import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as messagebox
from tkinter import font as tkfont
import colorsys


class ModernInputDialog:
    def __init__(self, iLayer):
        self.root = tk.Tk()
        self.root.title("Neural Network Quantization Parameters")
        self.iLayer = iLayer

        # Color scheme (soft and modern colors)
        self.colors = {
            'primary': '#4caf50',  # Modern green
            'secondary': '#388e3c',
            'background': '#f9f9f9',
            'card': '#ffffff',
            'text': '#333333',
            'subtext': '#777777',
            'border': '#dcdcdc',
            'hover': '#e8f5e9',
            'success': '#00c853',
            'error': '#f44336'
        }

        # Set up window
        self.setup_window()

        # Initialize variables
        self.setup_variables()

        # Create custom styles
        self.create_styles()

        # Create UI elements
        self.create_widgets()

        # Bind events
        self.bind_events()

    def setup_window(self):
        # Window size and position
        window_width = 600
        window_height = 800
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Window properties
        self.root.minsize(500, 400)
        self.root.configure(bg=self.colors['background'])

        # Enable window transparency
        self.root.attributes('-alpha', 0.0)
        self.fade_in()

    def fade_in(self):
        alpha = self.root.attributes('-alpha')
        if alpha < 1.0:
            alpha += 0.1
            self.root.attributes('-alpha', alpha)
            self.root.after(20, self.fade_in)

    def setup_variables(self):
        self.nBit_wx_var = tk.StringVar(value='8')
        self.nBit_A_var = tk.StringVar(value='8')
        self.nBit_act_var = tk.StringVar(value='8')
        self.status_var = tk.StringVar()

        # Store final values
        self.nBit_wx = 8
        self.nBit_A = 8
        self.nBit_act = 8

    def create_styles(self):
        style = ttk.Style()

        # Configure theme colors
        style.configure('Main.TFrame', background=self.colors['background'])
        style.configure('Card.TFrame', background=self.colors['card'])

        # Label styles for normal text
        style.configure('Title.TLabel',
                        font=('Microsoft YaHei UI', 20, 'bold'),
                        foreground=self.colors['text'],
                        background=self.colors['card'])

        style.configure('Subtitle.TLabel',
                        font=('Microsoft YaHei UI', 11),
                        foreground=self.colors['subtext'],
                        background=self.colors['card'])

        style.configure('Status.TLabel',
                        font=('Microsoft YaHei UI', 10),
                        foreground=self.colors['subtext'],
                        background=self.colors['background'])

        # Label styles for header (inverted colors)
        style.configure('HeaderTitle.TLabel',
                        font=('Microsoft YaHei UI', 20, 'bold'),
                        foreground='#ffffff',
                        background='transparent')

        style.configure('HeaderSubtitle.TLabel',
                        font=('Microsoft YaHei UI', 11),
                        foreground='#e0e0e0',
                        background='transparent')

        # Entry styles
        style.configure('Custom.TEntry',
                        fieldbackground=self.colors['background'],
                        borderwidth=2,
                        relief='solid')

        # Button styles
        style.configure('Primary.TButton',
                        font=('Microsoft YaHei UI', 11),
                        background=self.colors['primary'],
                        width=15)

        style.configure('Secondary.TButton',
                        font=('Microsoft YaHei UI', 11),
                        width=15)

        # Parameter card style
        style.configure('Param.TFrame',
                        background=self.colors['card'],
                        relief='raised',
                        borderwidth=1)

    def create_gradient_canvas(self, parent, color1, color2):
        canvas = tk.Canvas(
            parent,
            width=600,
            height=120,  # Increased height for title
            highlightthickness=0
        )

        # Create gradient effect
        for i in range(120):  # Increased to match new height
            # Calculate gradient color
            ratio = i / 120
            r1, g1, b1 = [int(color1[i:i + 2], 16) for i in (1, 3, 5)]
            r2, g2, b2 = [int(color2[i:i + 2], 16) for i in (1, 3, 5)]
            r = int(r1 * (1 - ratio) + r2 * ratio)
            g = int(g1 * (1 - ratio) + g2 * ratio)
            b = int(b1 * (1 - ratio) + b2 * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'

            canvas.create_line(0, i, 600, i, fill=color)

        return canvas

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Top gradient background with titles embedded
        gradient_frame = tk.Frame(main_frame, bg=self.colors['background'])
        gradient_frame.pack(fill=tk.X, pady=(0, 25))

        # Create gradient canvas
        canvas = self.create_gradient_canvas(
            gradient_frame,
            self.colors['primary'],
            self.colors['secondary']
        )
        canvas.pack(fill=tk.X)

        # Draw text directly on the canvas
        canvas.create_text(
            300, 40,  # X, Y coordinates
            text=f"Layer {self.iLayer + 1} Configuration",
            fill="#ffffff",
            font=('Microsoft YaHei UI', 20, 'bold')
        )

        canvas.create_text(
            300, 80,  # X, Y coordinates
            text="Please set the bit width for each quantization parameter",
            fill="#e0e0e0",
            font=('Microsoft YaHei UI', 11)
        )

        # Content area
        content_frame = ttk.Frame(main_frame, style='Card.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Input area
        self.create_input_section(content_frame)

        # Button area
        button_frame = ttk.Frame(content_frame, style='Card.TFrame')
        button_frame.pack(fill=tk.X, pady=20)

        ttk.Button(
            button_frame,
            text="Confirm",
            style='Primary.TButton',
            command=self.confirm
        ).pack(side=tk.RIGHT, padx=15)

        ttk.Button(
            button_frame,
            text="Cancel",
            style='Secondary.TButton',
            command=self.on_closing
        ).pack(side=tk.RIGHT, padx=5)

        # Status bar
        status_frame = ttk.Frame(main_frame, style='Main.TFrame')
        status_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style='Status.TLabel'
        ).pack(side=tk.LEFT)

    def create_input_section(self, parent):
        input_frame = ttk.Frame(parent, style='Card.TFrame')
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # Create input fields
        params = [
            ("LUT Cosine Bit Width", self.nBit_wx_var, "Quantization bits for cosine values in lookup table"),
            ("Amplitude Bit Width", self.nBit_A_var, "Quantization bits for signal amplitude"),
            ("Inter-layer Data Bit Width", self.nBit_act_var, "Quantization bits for data transfer between layers")
        ]

        for i, (label, var, tooltip) in enumerate(params):
            # Create a card-like frame for each parameter
            param_frame = ttk.Frame(input_frame, style='Param.TFrame')
            param_frame.pack(fill=tk.X, pady=15, ipady=10)

            # Add padding inside the parameter frame
            inner_frame = ttk.Frame(param_frame, style='Card.TFrame')
            inner_frame.pack(fill=tk.X, padx=15, pady=10)

            # Label with icon
            label_frame = ttk.Frame(inner_frame, style='Card.TFrame')
            label_frame.pack(fill=tk.X)

            # Parameter name with larger font
            param_label = ttk.Label(
                label_frame,
                text=label,
                font=('Microsoft YaHei UI', 12, 'bold'),
                foreground=self.colors['text'],
                background=self.colors['card']
            )
            param_label.pack(side=tk.LEFT, anchor='w')

            # Tooltip frame
            tooltip_frame = ttk.Frame(inner_frame, style='Card.TFrame')
            tooltip_frame.pack(fill=tk.X, pady=(5, 10))

            # Tooltip with icon simulation
            tooltip_text = f"ℹ {tooltip}"
            tooltip_label = ttk.Label(
                tooltip_frame,
                text=tooltip_text,
                font=('Microsoft YaHei UI', 9, 'italic'),
                foreground=self.colors['subtext'],
                background=self.colors['card']
            )
            tooltip_label.pack(side=tk.LEFT)

            # Entry field in its own frame
            entry_frame = ttk.Frame(inner_frame, style='Card.TFrame')
            entry_frame.pack(fill=tk.X)

            # Current value indicator
            value_label = ttk.Label(
                entry_frame,
                text="Value:",
                font=('Microsoft YaHei UI', 10),
                foreground=self.colors['text'],
                background=self.colors['card']
            )
            value_label.pack(side=tk.LEFT, padx=(0, 10))

            # Entry field
            entry = ttk.Entry(
                entry_frame,
                textvariable=var,
                width=10,
                style='Custom.TEntry',
                font=('Microsoft YaHei UI', 11)
            )
            entry.pack(side=tk.LEFT)

            # Bits indicator
            bits_label = ttk.Label(
                entry_frame,
                text="bits",
                font=('Microsoft YaHei UI', 10),
                foreground=self.colors['text'],
                background=self.colors['card']
            )
            bits_label.pack(side=tk.LEFT, padx=(5, 0))

            # Add validation
            vcmd = (self.root.register(self.validate_input), '%P')
            entry.configure(validate='key', validatecommand=vcmd)

    def bind_events(self):
        self.root.bind('<Return>', lambda e: self.confirm())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if messagebox.askokcancel("Confirm", "Are you sure you want to cancel?"):
            self.root.destroy()

    def validate_input(self, value):
        if value == "": return True
        try:
            num = int(value)
            return 0 < num <= 32
        except ValueError:
            return False

    def confirm(self):
        try:
            values = [
                int(self.nBit_wx_var.get()),
                int(self.nBit_A_var.get()),
                int(self.nBit_act_var.get())
            ]

            if not all(0 < x <= 32 for x in values):
                raise ValueError("Bit width must be between 1 and 32 bits")

            self.nBit_wx, self.nBit_A, self.nBit_act = values

            self.status_var.set("✓ Parameters set successfully. Window will close...")
            self.root.update()

            # Fade out effect
            self.fade_out()

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            self.status_var.set("⚠ Please check your input values")

    def fade_out(self):
        def fade():
            alpha = self.root.attributes('-alpha')
            if alpha > 0:
                alpha -= 0.1
                self.root.attributes('-alpha', alpha)
                self.root.after(20, fade)
            else:
                self.root.destroy()

        fade()

    def get_values(self):
        self.root.mainloop()
        return self.nBit_wx, self.nBit_A, self.nBit_act


class InitialChoiceDialog:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Quantization Configuration")
        self.choice = None

        # Color scheme
        self.colors = {
            'primary': '#4caf50',
            'secondary': '#388e3c',
            'background': '#f9f9f9',
            'text': '#ffffff'
        }

        # Center the window
        window_width, window_height = 600, 350
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg=self.colors['background'])

        # Gradient background
        gradient_frame = tk.Frame(self.root)
        gradient_frame.pack(fill=tk.X, pady=(0, 20))

        # Canvas for gradient
        canvas = tk.Canvas(
            gradient_frame,
            width=600,
            height=100,
            highlightthickness=0
        )
        canvas.pack()

        # Create gradient effect
        for i in range(100):
            ratio = i / 100
            r1, g1, b1 = [int(self.colors['primary'][i:i + 2], 16) for i in (1, 3, 5)]
            r2, g2, b2 = [int(self.colors['secondary'][i:i + 2], 16) for i in (1, 3, 5)]
            r = int(r1 * (1 - ratio) + r2 * ratio)
            g = int(g1 * (1 - ratio) + g2 * ratio)
            b = int(b1 * (1 - ratio) + b2 * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_line(0, i, 600, i, fill=color)

        # Draw title text directly on canvas
        canvas.create_text(
            300, 50,
            text="Quantization Configuration",
            font=('Microsoft YaHei UI', 18, 'bold'),
            fill=self.colors['text']
        )

        # Content area
        content_frame = tk.Frame(self.root, bg='#ffffff', bd=1, relief=tk.SOLID)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Create labels and buttons
        question_label = tk.Label(
            content_frame,
            text="Would you like to customize the quantization\nbit width for each layer?",
            font=('Microsoft YaHei UI', 12),
            bg='#ffffff',
            fg='#333333',
            pady=20
        )
        question_label.pack()

        button_frame = tk.Frame(content_frame, bg='#ffffff')
        button_frame.pack(pady=20)

        yes_button = tk.Button(
            button_frame,
            text="Yes",
            font=('Microsoft YaHei UI', 11),
            bg=self.colors['primary'],
            fg='white',
            command=self.on_yes,
            width=10,
            height=2,
            relief=tk.FLAT
        )
        yes_button.pack(side=tk.LEFT, padx=20)

        no_button = tk.Button(
            button_frame,
            text="No",
            font=('Microsoft YaHei UI', 11),
            bg=self.colors['secondary'],
            fg='white',
            command=self.on_no,
            width=10,
            height=2,
            relief=tk.FLAT
        )
        no_button.pack(side=tk.RIGHT, padx=20)

    def on_yes(self):
        self.choice = True
        self.root.destroy()

    def on_no(self):
        self.choice = False
        self.root.destroy()

    def get_choice(self):
        self.root.mainloop()
        return self.choice


# Test code
if __name__ == "__main__":
    dialog = ModernInputDialog(0)
    wx, a, act = dialog.get_values()
    print(f"Parameter values set: wx={wx}, A={a}, act={act}")