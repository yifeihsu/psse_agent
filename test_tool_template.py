import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-7B-Instruct")
conversation = [
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "name": "wls_from_path",
        "content": json.dumps({"success": True})
    }
]

prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
print("DUMP:\n" + prompt)
