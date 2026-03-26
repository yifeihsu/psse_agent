import json
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# Load the tokenizer used for evaluation
adapter_path = "outputs/gpt_oss_sft_power_agent/lora"
try:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
except Exception as e:
    print(f"Failed to load tokenizer from {adapter_path}: {e}")
    # try loading the base tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-7B-Instruct") # guess

# Load one test sample
test_file = "out_traces_balanced/sft_traces.test.jsonl"
try:
    with open(test_file, "r", encoding="utf-8") as f:
        test_sample = json.loads(f.readline())
except Exception as e:
    print(f"Failed to load {test_file}: {e}")
    test_sample = None

if test_sample:
    print("=== Eval Script Serialized Prompt (First 2 messages) ===")
    conversation = [test_sample[0], test_sample[1]]
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(prompt)
    print("\n" + "="*50 + "\n")

    print("=== Training Data Target Serialization (Including Assistant response) ===")
    target_prompt = tokenizer.apply_chat_template(
        test_sample,
        tokenize=False,
        add_generation_prompt=False,
    )
    print(target_prompt)
