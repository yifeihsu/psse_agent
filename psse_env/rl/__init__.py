"""On-policy reinforcement learning for the diagnostic agent (plan of 2026-10-13).

The package reuses what the research pipeline already has: the closed-loop
evaluator rolls the sampled policy out and audits every episode against the
offline truth, the SFT masks give the assistant-only token loss, PEFT holds
the adapter.  What is new is the reward contract (``reward``), the
group-relative advantage (``advantages``), the per-step training rows built
from the exact tokens the policy sampled (``rows``), the LoRA policy-gradient
update (``update``) and the iteration driver with its trajectory store
(``train``).  See docs/rl_experiments_20261014.md.
"""
