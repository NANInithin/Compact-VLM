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
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return "❌ Error: Image file not found."

    # --- THE FIX: DEFENSIVE SYSTEM PROMPT ---
    # We explicitly instruct the model to admit ignorance.
    system_instruction = (
        "You are a helpful visual assistant. "
        "Answer strictly based on what is visible in the image. "
        "If the object asked about is not visible, you must say 'I do not see that object'. "
        "Do not hallucinate or make up facts."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                # We combine the system instruction with the user question
                {"type": "text", "text": f"{system_instruction}\n\nQuestion: {question}"}
            ]
        },
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_texts[0].split("Assistant:")[-1].strip()

if __name__ == "__main__":
    processor, model = load_model()
    
    TEST_IMAGE = "data/banana_real.jpg" # Make sure this matches your filename
    
    print("\n" + "="*50)
    print("🛡️  THE DEFENSIVE PROMPT EXPERIMENT (The Fix)")
    print("="*50)

    # Trap 1: The Sticker
    q1 = "Read the text on the sticker on the banana."
    print(f"\n1️⃣  Question: {q1}")
    ans1 = run_inference(processor, model, TEST_IMAGE, q1)
    print(f"   🤖 Answer: {ans1}")

    # Trap 2: The Bowl
    q2 = "What material is the bowl made of?"
    print(f"\n2️⃣  Question: {q2}")
    ans2 = run_inference(processor, model, TEST_IMAGE, q2)
    print(f"   🤖 Answer: {ans2}")

    # Trap 3: The Apple
    q3 = "Describe the apple next to the banana."
    print(f"\n3️⃣  Question: {q3}")
    ans3 = run_inference(processor, model, TEST_IMAGE, q3)
    print(f"   🤖 Answer: {ans3}")

    print("\n" + "="*50)