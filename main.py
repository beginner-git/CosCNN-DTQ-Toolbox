# CosCNN-DTQ Toolbox
# Copyright (C) 2024-2025 Guoyang Liu
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

import subprocess
import os


def run_train():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the 'train' directory
    train_dir = os.path.join(current_dir, "train")

    if not os.path.exists(train_dir):
        print(f"Error: Directory {train_dir} does not exist!")
        return

    try:
        # Change to the train directory and run main.py
        result = subprocess.run(
            ["python", "main.py"],
            cwd=train_dir,  # Key: specify working directory as train
            check=True,
            capture_output=True,
            text=True
        )
        print("Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Execution failed with error message:")
        print(e.stderr)


if __name__ == '__main__':
    run_train()
