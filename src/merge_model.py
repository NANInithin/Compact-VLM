import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- CONFIGURATION ---
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ADAPTER_PATH = "smolvlm-finetuned-defense" # Your local adapter folder
OUTPUT_DIR = "smolvlm-defense-merged"

def merge():
    print("⬇️  Loading Base Model...")
    # Load base model in full precision (or bfloat16) to ensure clean merge
    # We cannot merge 4-bit (QLoRA) models directly easily, so we load base in bf16
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("🔗 Loading & Merging Adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    # The Magic Command: Merges weights and removes the adapter wrapper
    model = model.merge_and_unload()
    
    print(f"💾 Saving Merged Model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    
    # Don't forget the processor!
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("✅ Done! You now have a standalone model.")

if __name__ == "__main__":
    merge()