# Preliminary E2B small closed-loop evaluation

Date: 2026-08-22

## Scope

This is a diagnostic-only, matched five-episode evaluation. It is not a release result and does not replace the frozen full-study evaluation.

- Source commit: `971f72cccae445c3bf49cf9b3752a536e279686a`
- Model: `unsloth/gemma-4-E2B-it`
- Pinned model revision: `f0c5915`
- Dataset receipt SHA-256: `2a1f36bd0fe9e192b5aa53b8cc48791840b1d4d7aba1d6e28bb5a493bd7a3b54`
- Suite SHA-256: `d9248ebcec66799652361a12290f546813e1f8b2f4879b56603a8b435bdff28d`
- Protocol: canonical
- Evaluation seed: `20260821`
- Episodes: 5, one each for efficiency, forced-error recovery, invalid-action recovery, partial-success retention, and standard success
- Maximum steps per episode: 8
- Decoding: greedy (`do_sample=False`, temperature 0), maximum 64 new tokens
- Registered tools: 18

The two policies used the same suite bytes and deterministic decoding. DAgger ran on an RTX Pro 6000; the final matched BC0 run ran on an L40S. That hardware difference affects runtime but not the deterministic model inputs, weights, or decoding configuration.

## Results

| Metric | BC0 | DAgger continuation |
| --- | ---: | ---: |
| Aggregate score | 0 | 0 |
| Resolved episodes | 0/5 | 0/5 |
| Terminal episodes | 0/5 | 0/5 |
| Loop episodes | 5/5 | 5/5 |
| Invalid actions | 10 | 13 |
| Episodes with invalid actions | 5/5 | 5/5 |
| Control-quarantined episodes | 5/5 | 5/5 |
| Total steps | 16 | 19 |
| Successful usable tool calls | 0 | 0 |
| WLS calls | 0 | 0 |
| Specialized tool calls | 0 | 0 |
| Final held-out validation loss | 1.9485595 | 1.9607674 |

The DAgger continuation made one syntactically valid `get_measurement_context` call, but its state reference did not match a known controller state, so the call was not usable. Most generations were free-form or malformed, often reaching the 64-token limit without producing a valid canonical tool call. Other short generations used `state_id: episode`, which the alias bridge correctly rejected as unknown.

The preliminary result is therefore negative: this continuation did not improve closed-loop behavior over BC0. It slightly increased invalid actions and its final validation loss was about 0.63% higher than BC0.

## Artifact integrity

- `bc0_balanced.json`: `b32957022f6473f984444149a54cb9c4d9aa4e055910338466eaf3e066e497bb`
- `dagger_balanced.json`: `1ed531440ea7b22b72d47f510f5b64f0c1f7951999c2cca919d22976c0125ef6`
- `small_balanced_suite.json`: `d9248ebcec66799652361a12290f546813e1f8b2f4879b56603a8b435bdff28d`
- `bc0_stage_receipt.json`: `330242fa63b3b453dffaa161671824e2f3f2355166b451b2e495df71fba2a5f0`
- `dagger_stage_receipt.json`: `45bd49d2a5e8fe38ca294d04f81b815f57e0ba4b7d844db1f375548a867d14b2`

The local copies were checked byte-for-byte against the remote evaluation artifacts.

## Next gate

Before spending time on a larger model or full closed-loop study, train a bounded syntax-focused continuation using only training roots. Require complete canonical tool calls, valid controller state references, and termination within the decoding budget. Rerun this frozen five-root diagnostic suite only after an offline tool-call validity gate passes.
