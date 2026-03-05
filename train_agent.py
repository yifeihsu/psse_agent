import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
# For RTX 4080 (16GB VRAM), training a 20B parameter model requires 4-bit config 
# For RTX 4080 (16GB VRAM), we use the 8B parameter model in 4-bit config.
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit" # Fits gracefully in 16GB
MAX_SEQ_LENGTH = 2048 # Adjust down if OOM occurs
DATASET_PATH = "data/split_train.jsonl"
OUTPUT_DIR = "outputs/gpt_oss_agent"

# Get Unsloth optimized model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # Auto-detect
    load_in_4bit = True, # Crucial for 16GB VRAM
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
# We simply rely on Unsloth loading the tokenizer directly, no get_chat_template needed
# tokenizer = get_chat_template(...)

def format_openai_tools_for_chatml(convo):
    # ChatML template usually expects string content for every role
    # We will format this directly into strings to avoid Jinja exceptions
    text = ""
    for msg in convo:
        role = msg.get("role", "user")
        
        # If assistant has tool calls but no content, serialize the tool call to content
        if role == "assistant" and "tool_calls" in msg and not msg.get("content"):
            tool_call = msg["tool_calls"][0]
            func = tool_call["function"]
            content = f'<tool_call>\n{{"name": "{func["name"]}", "arguments": {func["arguments"]}}}\n</tool_call>'
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
        # If this is the tool's return value
        elif role == "tool" or ("tool_call_id" in msg and "name" in msg):
            content = f'<tool_response>\n{{"name": "{msg.get("name", "")}", "content": {msg.get("content", "")}}}\n</tool_response>'
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
            
        else:
            # Standard message
            content = msg.get("content")
            if content is None:
                content = ""
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
    # Add generation prompt for the final assistant response
    text += "<|im_start|>assistant\n"
    return text

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = []
    for idx, convo in enumerate(convos):
        chat_str = format_openai_tools_for_chatml(convo)
        texts.append(chat_str)
        if idx == 0:
            print("================ FIRST TRACE PARSED ================\n")
            print(chat_str)
            print("\n====================================================\n")
    return { "text" : texts }

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
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # Set to True for speed, but False is safer for VRAM spikes
    args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 1, # Minimal VRAM usage
        gradient_accumulation_steps = 4, # Effective batch size = 4
        warmup_steps = 5,
        max_steps = 100, # Use `num_train_epochs = 1` for full run
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(), # RTX 4080 supports full bfloat16 natively
        logging_steps = 1,
        optim = "adamw_8bit", # 8-bit optimizer saves VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Disable wandb for local run
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
