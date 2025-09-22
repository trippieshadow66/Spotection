# test_env.py
# Quick check that all dependencies are installed and working

import flask
import cv2
import numpy as np
import torch
import ultralytics

def main():
    print("✅ Environment check starting...\n")

    print(f"Flask version: {flask.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Numpy version: {np.__version__}")
    print(f"Torch version: {torch.__version__}")
    print(f"Ultralytics (YOLO) version: {ultralytics.__version__}")

    # Check CUDA (GPU support) for PyTorch
    print("\nPyTorch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    print("\n All imports succeeded!")

if __name__ == "__main__":
    main()
