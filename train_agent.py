import unsloth  # must be first

import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
MODEL_NAME = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DATASET_PATH = "data/split_train.jsonl"
OUTPUT_DIR = "outputs/gpt_oss_agent"

# Get Unsloth optimized model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
    full_finetuning = False,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized
    bias = "none",    # Optimized
    use_gradient_checkpointing = "unsloth", # Crucial for VRAM savings
    random_state = 3407,
)

# ==========================================
# 2. Data Preparation
# ==========================================

def formatting_prompts_func(examples):
    texts = []
    for convo in examples["messages"]:
        # Strip None-valued keys added by Arrow schema unification
        clean = [{k: v for k, v in m.items() if v is not None} for m in convo]
        texts.append(tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=False))
    return {"text": texts}

print(f"Loading dataset from {DATASET_PATH}")
dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")

# Apply formatting
dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 3. Trainer Setup (Optimized for RTX 4080)
# ==========================================
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,  # set num_train_epochs=1 for full run
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        dataset_text_field = "text",
        dataset_num_proc = 1,
        max_length = MAX_SEQ_LENGTH,
        packing = False,
        report_to = "none",
    ),
)

# ==========================================
# 4. Training
# ==========================================
print("Starting training...")
trainer_stats = trainer.train()
print(f"Training memory stats: \n{trainer_stats.metrics}")

# ==========================================
# 5. Save the Merged Model
# ==========================================
# Save the LoRA adapters
model.save_pretrained(f"{OUTPUT_DIR}/lora_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_model")

print("Training script finished successfully. The LoRA adapters are saved.")
# To save as GGUF (for ollama/llama.cpp):
# print("Exporting to GGUF (q4_k_m)...")
# model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
print("Done!")
