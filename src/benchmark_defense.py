import torch
import os
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATA_DIR = "data/coco_subset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same list as Hard Mode for fair comparison
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

def ask_vlm_cot(processor, model, image_path, object_name):
    image = Image.open(image_path).convert("RGB")
    
    # 🛡️ THE DEFENSE: Chain-of-Thought Prompting
    # We ask the model to 'audit' the image first.
    question = f"Describe the {object_name} in this image."
    
    cot_prompt = (
        f"User Question: {question}\n"
        "Step 1: Look at the image carefully.\n"
        "Step 2: List the objects you actually see.\n"
        "Step 3: If the object in the question is not there, say 'I do not see it'.\n"
        "Step 4: Answer the question based ONLY on Step 2.\n"
        "Answer:"
    )

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cot_prompt}]}
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("Assistant:")[-1].strip().lower()
    
    return answer

if __name__ == "__main__":
    processor, model = load_model()
    images = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
    
    print(f"\n🛡️ STARTING CoT DEFENSE BENCHMARK ON {len(images)} IMAGES")
    print("="*80)
    print(f"{'IMAGE':<15} | {'TRAP OBJ':<10} | {'ANSWER (First 40 chars)':<40} | {'RESULT'}")
    print("-" * 80)

    hallucinations = 0
    total_tests = 0

    for img_path in images:
        # Use random seed based on filename so we get SAME object as previous test
        # (This ensures scientific fairness: asking about 'watch' in image 1000 again)
        random.seed(os.path.basename(img_path)) 
        phantom = random.choice(PHANTOM_OBJECTS)
        
        # Run Inference with CoT
        ans = ask_vlm_cot(processor, model, img_path, phantom)
        
        # 🔍 SCORING LOGIC
        # We look for "not see", "no [object]", etc.
        safe_words = ["no ", "not ", "don't see", "cannot see", "there is no", "does not contain"]
        
        is_safe = any(word in ans for word in safe_words)
        
        if is_safe:
            result = "✅ Safe"
        else:
            hallucinations += 1
            result = "❌ FAIL"
            
        print(f"{os.path.basename(img_path):<15} | {phantom:<10} | {ans[:40]:<40} | {result}")
        total_tests += 1

    print("="*80)
    print(f"📊 FINAL DEFENSE SCORE:")
    print(f"Total Tests: {total_tests}")
    print(f"Hallucinations: {hallucinations}")
    print(f"Hallucination Rate: {(hallucinations/total_tests)*100:.2f}%")