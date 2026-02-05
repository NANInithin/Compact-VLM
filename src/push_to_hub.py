import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

# --- CONFIGURATION ---
HF_USERNAME = "NANI-Nithin"  # <--- CHANGE THIS
REPO_NAME = "SmolVLM-Hallucination-Defense"
LOCAL_ADAPTER_PATH = "smolvlm-finetuned-defense"
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

def push_model():
    print(f"🚀 Pushing adapter to {HF_USERNAME}/{REPO_NAME}...")
    
    # 1. Load the Adapter locally
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.bfloat16,
        device_map="cpu" # Load on CPU just for uploading
    )
    model = PeftModel.from_pretrained(model, LOCAL_ADAPTER_PATH)
    
    # 2. Push Adapter to Hub
    model.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}", use_auth_token=True)
    
    # 3. Push Processor (for ease of use)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    processor.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}", use_auth_token=True)
    
    print("✅ Upload Complete! Check your profile.")

if __name__ == "__main__":
    push_model()