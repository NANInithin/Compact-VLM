import torch
import json
import random
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image

# --- CONFIGURATION ---
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ADAPTER_PATH = "smolvlm-finetuned-defense"
TRAIN_DATA_FILE = "data/mixed_training_data.json"
DEVICE = "cuda"

def load_model():
    print("⬇️  Loading Model for Inspection...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, 
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    # Force Vision Tower to BFloat16
    model.model.vision_model.to(torch.bfloat16)
    
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    return processor, model

def test_examples():
    processor, model = load_model()
    
    with open(TRAIN_DATA_FILE, "r") as f:
        data = json.load(f)
        
    # Separate the data into "Traps" and "Real"
    traps = [d for d in data if "I do not see" in d["conversations"][1]["content"][0]["text"]]
    reals = [d for d in data if "I do not see" not in d["conversations"][1]["content"][0]["text"]]
    
    print(f"\n🧪 CHECKING MODEL BEHAVIOR (Total Data: {len(data)})")
    
    # 1. Test 3 Traps
    print("\n💀 TRAP TEST (Expectation: Refusal)")
    print("-" * 60)
    for i in range(min(3, len(traps))):
        example = traps[i]
        run_inference(processor, model, example)

    # 2. Test 3 Reals
    print("\n🍎 REAL TEST (Expectation: Description)")
    print("-" * 60)
    for i in range(min(3, len(reals))):
        example = reals[i]
        run_inference(processor, model, example)

def run_inference(processor, model, example):
    image_path = example["image"]
    question = example["conversations"][0]["content"][1]["text"]
    expected = example["conversations"][1]["content"][0]["text"]
    
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("Assistant:")[-1].strip()
    
    print(f"Q: {question}")
    print(f"Expect: {expected[:50]}...")
    print(f"Model:  {answer}")
    print("-" * 30)

if __name__ == "__main__":
    test_examples()