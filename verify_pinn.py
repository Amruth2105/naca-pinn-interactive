import sys
sys.path.append('c:/Users/amrut/.gemini/antigravity/scratch/pinn_naca6412')
import pinn_lib
import torch

def test():
    print("Testing PINN Lib...")
    
    # 1. Dataset
    print("Generating dataset...")
    X_b, Y_b, U_b, V_b, X_c, Y_c, _, _ = pinn_lib.get_dataset("6412", 100, 100)
    assert isinstance(X_b, torch.Tensor)
    print("Dataset generated successfully.")
    
    # 2. Model
    print("Initializing model...")
    model = pinn_lib.PINN()
    
    # 3. Forward Pass
    print("Running forward pass...")
    out = model(X_b, Y_b)
    assert out.shape == (100, 3) # u, v, p for 3 walls + 25 airfoil? Wait, 100 total
    print("Forward pass successful.")
    
    print("All checks passed.")

if __name__ == "__main__":
    test()
