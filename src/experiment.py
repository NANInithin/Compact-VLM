import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

def load_model():
    print(f"⬇️  Loading model: {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        _attn_implementation="sdpa" if DEVICE == "cuda" else "eager"
    ).to(DEVICE)
    return processor, model

def run_inference(processor, model, image_path, question):
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        },
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Clean up the response to just get the answer
    answer = generated_texts[0].split("Assistant:")[-1].strip()
    return answer

if __name__ == "__main__":
    processor, model = load_model()
    
    print("\n" + "="*40)
    print("🧪  THE PURPLE BANANA EXPERIMENT")
    print("="*40)

    # Test 1: Control (Yellow)
    print("\n1️⃣  Testing Control (Yellow Banana)...")
    ans1 = run_inference(processor, model, "data/banana_real.jpg", "What color is the banana?")
    print(f"   🤖 Answer: {ans1}")

    # Test 2: Trap (Purple)
    print("\n2️⃣  Testing Trap (Purple Banana)...")
    ans2 = run_inference(processor, model, "data/banana_purple_real.png", "What color is the banana?")
    print(f"   🤖 Answer: {ans2}")
    
    print("\n" + "="*40)
    if "yellow" in ans2.lower():
        print("🚨 RESULT: FAILURE (Modality Collapse detected!)")
        print("   The model ignored the image and guessed 'Yellow' based on text.")
    else:
        print("🎉 RESULT: SUCCESS (Robust Vision)")
        print("   The model correctly saw the unusual color.")