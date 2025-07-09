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
