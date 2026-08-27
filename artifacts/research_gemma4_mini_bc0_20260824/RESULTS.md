# Gemma 4 12B mini-BC0 result

- Source commit: `32f67f8be4b1538833e1c9278faadc2db3dd5d45`
- Training job: `16297675` (all 32 optimizer steps completed; its original postflight rejected a semantically plausible but non-identical free-text argument)
- Finalization job: `16299654` (`COMPLETED`, exit `0:0`, 4m05s)
- Finalization reused the saved adapter (`resumed_saved_adapter_finalization: true`); it did not repeat training.
- Model: `google/gemma-4-12B-it@707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, 4-bit base plus rank-16 LoRA.
- Split: 128 training rows / 88 roots; 32 validation rows / 7 roots; zero root overlap.
- Training: 32 steps, checkpoints at 8/16/24/32, train loss `0.18198618275346234`.
- Validation losses: step 8 `0.3005967140197754`; step 16 `0.13056260347366333`; step 24 `0.08849798887968063`; step 32 `0.07105836272239685`.
- Reload canary: fresh base reconstructed, adapter reloaded, exactly one tool call parsed, expected/generated tool both `ask_for_more_evidence`; exact free-text action match was false and remains recorded as a diagnostic.
- Final research postflight: passed.
- Release eligibility: false. This is an integration mini-BC0 artifact, not a closed-loop semantic-quality result.

Remote adapter and checkpoints remain under:

`/scratch/yx3882/research_gemma4_small_20260824_fe94580/bc0/mini`

## SHA-256 (remote/local matched)

```text
d558653bdd37758449787a78fb65b49419338d8799d20e29651bd6bf74056b20  research_run.json
75d4a400f2dbcbcea8e16b480511942530e3f4194c8c91e58c4fabfd2e48b057  training_stage.json
0648e60a06c4f2eeaba1e03f66e21409caa211aa3d693836413b40b4622d5458  trainer_state.json
16aa34002982c68609767404b4e5d49b8443843bf321dee0b13c3e4c8d6aeab9  mini_bc0_postflight.json
```
