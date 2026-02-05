import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from PIL import Image

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATA_FILE = "data/mixed_training_data.json"
OUTPUT_DIR = "smolvlm-finetuned-defense"
DEVICE = "cuda"

def train():
    print(f"⬇️  Loading Model in 4-bit (BFloat16 Mode)...")
    
    # 1. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, 
    )

    # 2. Load Base Model
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # --- FIX 1: DISABLE CACHE & PREPARE FOR KBIT TRAINING ---
    model.config.use_cache = False  # Required for gradient checkpointing
    model.gradient_checkpointing_enable() 
    
    # This helper function stabilizes the layers (casts LayerNorm to fp32, freezes vision, etc.)
    model = prepare_model_for_kbit_training(model)
    
    # --- FIX 2: FORCE VISION TOWER TO BFLOAT16 (HARD CAST) ---
    # Even after preparation, we explicitly force the vision model to bf16 
    # to prevent the "Float32 source" error.
    model.model.vision_model.to(torch.bfloat16)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 3. LoRA Config
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM" 
    )

    # 4. Load Dataset
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # 5. Data Collator
    def collate_fn(examples):
        texts = [processor.apply_chat_template(e["conversations"], tokenize=False) for e in examples]
        images = [Image.open(e["image"]).convert("RGB") for e in examples]
        
        batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
        
        # Ensure pixel values are BFloat16 to match the Vision Tower
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

        batch["labels"] = batch["input_ids"].clone()
        return batch

    # 6. Training Arguments
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        bf16=True,                # Enable BFloat16
        fp16=False,               # Disable FP16
        logging_steps=5,
        save_strategy="epoch",
        max_length=512,
        packing=False,
        remove_unused_columns=False,
        # Reduce memory fragmentation
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False} 
    )

    print("🚀 Starting Training... (Monitor your GPU Task Manager)")
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=collate_fn, 
    )

    trainer.train()
    
    print("✅ Training Complete! Saving adapter...")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    train()