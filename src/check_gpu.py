import sys
import torch

def diagnose_gpu():
    print("=" * 60)
    print("            SYSTEM & PYTORCH DIAGNOSTICS            ")
    print("=" * 60)
    
    # 1. Software Environment
    print(f"Python Version:  {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 2. CUDA Compilation & Driver Details
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")
    
    if not cuda_available:
        print("\n[!] ERROR: PyTorch cannot see any CUDA GPUs.")
        print("This usually means a driver/toolkit version mismatch.")
        try:
            # Try to get the driver version PyTorch sees before failing
            driver_ver = torch.cuda.get_device_caption()
            print(f"CUDA Driver reported by system: {driver_ver}")
        except Exception:
            pass
        print("=" * 60)
        return

    print(f"PyTorch CUDA Compile Version: {torch.version.cuda}")
    
    # 3. Device Diagnostics
    device_count = torch.cuda.device_count()
    print(f"Detected GPU Count:           {device_count}")
    print("-" * 60)
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / (1024 ** 3)
        
        print(f"GPU [{i}]: {props.name}")
        print(f"  • Compute Capability: {props.major}.{props.minor}")
        print(f"  • Total VRAM:         {total_memory_gb:.2f} GB")
        
        # Test a minor tensor operation to ensure the driver handshake works
        try:
            test_tensor = torch.randn(1, 1).cuda(i)
            print(f"  • CUDA Handshake:     SUCCESS (Tensor initialized on GPU)")
            del test_tensor
        except Exception as e:
            print(f"  • CUDA Handshake:     FAILED! Error: {e}")
            
    print("=" * 60)

if __name__ == "__main__":
    diagnose_gpu()