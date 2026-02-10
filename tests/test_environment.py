import sys
import os

# 1. Add the project root to the path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports(): #commenting this out eventually to see if this cuts on testing time
    """
    Can we import the critical libraries?
    """
    try:
        import torch
        import cv2
        import numpy
        import ultralytics
        import scipy
        print("Core libraries imported successfully")
    except ImportError as e:
        print(f" Library Import Failed: {e}")
        sys.exit(1)

def test_gpu_availability():
    """
    Is the GPU actually visible to PyTorch?
    """
    import torch
    if torch.cuda.is_available():
        print(f" GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print(" No GPU detected. Running on CPU (Slower).")
        # We don't exit(1) here because sometimes you develop on a laptop without a GPU.

def test_src_access():
    """
    Can python see our local 'src' folder?
    """
    try:
        from src.skin_tone_classify import pandas
        # from src.audit_pipeline import run_audit # Uncomment once you write this
        print("'src' package is accessible.")
    except ImportError as e:
        print(f" Failed to import 'src' modules: {e}")
        print("   Did you run 'pip install -e .' or set PYTHONPATH?")
        sys.exit(1)

if __name__ == "__main__":
    print("--- Starting Environment Sanity Check ---")
    test_imports()
    test_src_access()
    test_gpu_availability()
    print("--- Check Complete: READY TO CODE ---")