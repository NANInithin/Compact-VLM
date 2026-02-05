import torch
import os
import json
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATA_DIR = "data/coco_subset"
OUTPUT_FILE = "data/mixed_training_data.json"
DEVICE = "cuda"

# Objects that are LIKELY NOT in the images (Traps)
PHANTOM_OBJECTS = [
    "toaster", "hair drier", "teddy bear", "microwave", "toothbrush", 
    "snowboard", "giraffe", "parking meter", "watch", "clock", "lamp", 
    "book", "bottle", "cup", "chair", "keyboard", "mouse", "stapler"
]

def load_model():
    print("⬇️  Loading Base Model for Data Generation...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return processor, model

def generate_positive_example(processor, model, image_path):
    # Ask the model what it sees to create a "Real" example
    image = Image.open(image_path).convert("RGB")
    question = "What is the main object in this image? Answer in 1-2 words."
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(**inputs, max_new_tokens=10)
    real_obj = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("Assistant:")[-1].strip().lower()
    
    # Now ask it to describe that real object
    desc_q = f"Describe the {real_obj}."
    messages[0]["content"][1]["text"] = desc_q
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("Assistant:")[-1].strip()
    
    return real_obj, description

def create_dataset():
    processor, model = load_model()
    images = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
    dataset = []

    print(f"🔨 Generating Mixed Dataset (Positive + Negative) from {len(images)} images...")

    for img_file in images:
        img_path = os.path.join(DATA_DIR, img_file)
        
        # 1. Generate ONE Positive Example (The Anchor)
        try:
            real_obj, real_desc = generate_positive_example(processor, model, img_path)
            dataset.append({
                "image": img_path,
                "conversations": [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Describe the {real_obj} in this image."}]},
                    {"role": "assistant", "content": [{"type": "text", "text": real_desc}]}
                ]
            })
            print(f"   [+] Added Real: {real_obj}")
        except:
            print(f"   [-] Skipped Real generation for {img_file}")

        # 2. Generate ONE Negative Example (The Defense)
        phantom = random.choice(PHANTOM_OBJECTS)
        dataset.append({
            "image": img_path,
            "conversations": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Describe the {phantom} in this image."}]},
                {"role": "assistant", "content": [{"type": "text", "text": f"I do not see a {phantom} in this image."}]}
            ]
        })
        print(f"   [-] Added Trap: {phantom}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Created {len(dataset)} balanced training examples in {OUTPUT_FILE}")

if __name__ == "__main__":
    create_dataset()