"""
Test script to verify PyTorch installation and CUDA availability
"""
import sys

def test_torch_installation():
    try:
        import torch
        print(f"✓ PyTorch installed successfully: {torch.__version__}")
        
        # Test CUDA availability
        if hasattr(torch, 'cuda'):
            if torch.cuda.is_available():
                print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
                print(f"✓ CUDA version: {torch.version.cuda}")
            else:
                print("⚠ CUDA is not available, will use CPU")
        else:
            print("✗ torch.cuda module not found - PyTorch installation is corrupted")
            return False
            
        # Test basic tensor operations
        x = torch.randn(3, 3)
        print(f"✓ Basic tensor operations work: {x.shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def test_numpy_installation():
    try:
        import numpy as np
        print(f"✓ NumPy installed successfully: {np.__version__}")
        
        # Test basic operations
        arr = np.array([1, 2, 3])
        print(f"✓ Basic NumPy operations work: {arr.shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    except Exception as e:
        print(f"✗ NumPy test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing PyTorch and NumPy installation...")
    print("=" * 50)
    
    numpy_ok = test_numpy_installation()
    torch_ok = test_torch_installation()
    
    print("=" * 50)
    if numpy_ok and torch_ok:
        print("✓ All tests passed! Environment is ready.")
    else:
        print("✗ Some tests failed. Please check the installation.")