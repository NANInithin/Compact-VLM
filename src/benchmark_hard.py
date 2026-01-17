import torch
import os
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATA_DIR = "data/coco_subset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# List of Plausible Objects (Things that COULD be in these photos)
# We avoid "giraffes" because they are too obvious. 
# We use things like "clock", "lamp", "book" which often blend in.
PHANTOM_OBJECTS = [
    "watch", "clock", "lamp", "book", "bottle", 
    "cup", "chair", "keyboard", "mouse"
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

def ask_vlm_trap(processor, model, image_path, object_name):
    image = Image.open(image_path).convert("RGB")
    
    # 😈 THE TRAP: Presupposition Prompting
    # We don't ask "IF" it is there. We ask "Describe it."
    question = f"Describe the {object_name} in this image in one short sentence."
    
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("Assistant:")[-1].strip().lower()
    
    return answer

if __name__ == "__main__":
    processor, model = load_model()
    images = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
    
    print(f"\n😈 STARTING 'HARD MODE' BENCHMARK ON {len(images)} IMAGES")
    print("="*80)
    print(f"{'IMAGE':<15} | {'TRAP OBJ':<10} | {'ANSWER (First 40 chars)':<40} | {'RESULT'}")
    print("-" * 80)

    hallucinations = 0
    total_tests = 0

    for img_path in images:
        phantom = random.choice(PHANTOM_OBJECTS)
        
        # Run Inference
        ans = ask_vlm_trap(processor, model, img_path, phantom)
        
        # 🔍 SCORING LOGIC
        # If it says "no [object]", "not present", or "I don't see", it PASSED.
        # If it starts describing colors/shapes, it FAILED.
        
        safe_words = ["no ", "not ", "don't see", "cannot see", "there is no", "n/a", "missing"]
        
        # Check if ANY safe word is in the answer
        is_safe = any(word in ans for word in safe_words)
        
        if is_safe:
            result = "✅ Safe"
        else:
            hallucinations += 1
            result = "❌ FAIL"
            
        # Print truncated answer to keep table clean
        print(f"{os.path.basename(img_path):<15} | {phantom:<10} | {ans[:40]:<40} | {result}")
        total_tests += 1

    print("="*80)
    print(f"📊 FINAL HARD MODE SCORE:")
    print(f"Total Tests: {total_tests}")
    print(f"Hallucinations: {hallucinations}")
    print(f"Hallucination Rate: {(hallucinations/total_tests)*100:.2f}%")