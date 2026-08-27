---
library_name: peft
model_name: bc0
tags:
- base_model:adapter:/scratch/yx3882/.cache/huggingface/hub/models--unsloth--gemma-4-12B-it/snapshots/55cdba0740a9765956f49501f689a66b098feda3
- lora
- sft
- transformers
- trl
licence: license
base_model: /scratch/yx3882/.cache/huggingface/hub/models--unsloth--gemma-4-12B-it/snapshots/55cdba0740a9765956f49501f689a66b098feda3
pipeline_tag: text-generation
---

# Model Card for bc0

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with SFT.

### Framework versions

- PEFT 0.20.0
- TRL: 1.10.0
- Transformers: 5.15.1
- Pytorch: 2.10.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```