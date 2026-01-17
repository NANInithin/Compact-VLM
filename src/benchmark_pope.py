import torch
import os
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATA_DIR = "data/coco_subset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# List of "Phantom Objects" to test (Objects unlikely to be in random photos)
PHANTOM_OBJECTS = [
    "toaster", "hair drier", "teddy bear", "microwave", "toothbrush", 
    "snowboard", "giraffe", "parking meter"
]

def load_model():
    print("⬇️  Loading Model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32, 
        _attn_implementation="sdpa" if DEVICE == "cuda" else "eager",
        local_files_only=True
    ).to(DEVICE)
    return processor, model

def ask_vlm(processor, model, image_path, object_name):
    image = Image.open(image_path).convert("RGB")
    
    # The "POPE" Prompt (Binary Yes/No)
    question = f"Is there a {object_name} in this image? Answer only Yes or No."
    
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("Assistant:")[-1].strip().lower()
    
    return answer

if __name__ == "__main__":
    processor, model = load_model()
    images = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
    
    print(f"\n🚀 STARTING BENCHMARK ON {len(images)} IMAGES")
    print("="*60)
    print(f"{'IMAGE':<20} | {'PHANTOM OBJ':<15} | {'ANSWER':<10} | {'RESULT'}")
    print("-" * 60)

    hallucinations = 0
    total_tests = 0

    for img_path in images:
        # Pick a random phantom object for this image
        phantom = random.choice(PHANTOM_OBJECTS)
        
        # Run Inference
        ans = ask_vlm(processor, model, img_path, phantom)
        
        # Check Result (Ideally, answer should be 'no')
        is_hallucination = "yes" in ans
        if is_hallucination:
            hallucinations += 1
            result = "❌ HALLUCINATION"
        else:
            result = "✅ Safe"
            
        print(f"{os.path.basename(img_path):<20} | {phantom:<15} | {ans:<10} | {result}")
        total_tests += 1

    print("="*60)
    print(f"📊 FINAL SCORE:")
    print(f"Total Tests: {total_tests}")
    print(f"Hallucinations: {hallucinations}")
    print(f"Hallucination Rate: {(hallucinations/total_tests)*100:.2f}%")