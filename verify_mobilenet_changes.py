
import argparse
import torch
import sys
import os

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from reid.models.rehearser import KernelLearning
from continual_train import parser

def test_config():
    print("Testing Argument Parser...")
    # Test default
    args_default = parser.parse_args([])
    assert args_default.mobilenet_type == 'small'
    print("Default argument check passed: mobilenet_type='small'")

    # Test custom
    args_large = parser.parse_args(['--mobilenet_type', 'large'])
    assert args_large.mobilenet_type == 'large'
    print("Custom argument check passed: mobilenet_type='large'")

def test_model_init():
    print("\nTesting Model Initialization...")
    
    # Test Small
    print("Initializing MobileNetV3 Small...")
    model_small = KernelLearning(n_kernel=1, model='mobile-v3', mobilenet_type='small')
    # Check output dimension or structure if possible, usually small is 576
    # Accessing internal backbone features to verify might be tricky without forward pass or inspecting non-public attributes easily, 
    # but we can check if it runs without error.
    print("Small model initialized successfully.")

    # Test Large
    print("Initializing MobileNetV3 Large...")
    model_large = KernelLearning(n_kernel=1, model='mobile-v3', mobilenet_type='large')
    print("Large model initialized successfully.")

if __name__ == "__main__":
    try:
        test_config()
        test_model_init()
        print("\nAll verifications passed!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        sys.exit(1)
