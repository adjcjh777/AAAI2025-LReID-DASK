
import sys
import os
import torch
import argparse

sys.path.append(os.getcwd())

from reid.models.resnet import make_model

def verify_attention():
    print("Verifying Attention Mechanism Integration...")
    
    # Mock arguments
    class Args:
        MODEL = '50x'
        dropout = 0.5
        with_attention = True
        
    args = Args()
    
    try:
        model = make_model(args, num_class=100, camera_num=6, view_num=0, pretrain=True)
        print("Model initialized successfully.")
        
        # Check if CBAM is present in the model
        has_cbam = False
        for name, module in model.named_modules():
            if 'cbam' in name:
                has_cbam = True
                print(f"Found CBAM module: {name}")
                break
                
        if has_cbam:
            print("VERIFICATION PASSED: CBAM module found in the model.")
        else:
            print("VERIFICATION FAILED: CBAM module NOT found in the model.")
            
    except Exception as e:
        print(f"VERIFICATION FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_attention()
