import torch

def check_gpu():
    print("Checking system...")
    if torch.cuda.is_available():
        print(f"✅ Success! CUDA is available.")
        print(f"   GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)} GB")
    else:
        print("❌ Error: CUDA not found. You are running on CPU.")

if __name__ == "__main__":
    check_gpu()