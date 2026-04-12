# QC report for updated traces

Dataset inspected: `sft_traces.sample5_per_category.jsonl`

## High-level verdict

These traces are **substantially better** than the previous generation. The biggest earlier problems—schema drift, synthetic follow-up user turns, repeated full numeric payloads in tool arguments, and visible absolute filesystem paths—are largely fixed.

## What is clearly improved

- 25 traces total, balanced as 5 each for:
  - `harmonic_anomaly`
  - `measurement_error`
  - `no_error`
  - `parameter_error`
  - `topology_error`
- Every trace begins with `wls_from_path`.
- There is exactly **one visible user turn per trace**.
- The middle steps are now assistant tool calls plus tool replies, which is the right structure for tool-calling SFT.
- Visible tool arguments are compact:
  - first `wls_from_path` call is always `{"case_path":"case14"}`
  - helper tools are compact
  - correction tools no longer carry giant arrays in visible arguments
- No visible absolute Windows paths remain.
- Hidden `runtime_context` now carries:
  - `case_aliases`
  - full helper payloads in `tool_context`
- Visible helper tools are present exactly where expected:
  - `get_parameter_context`
  - `get_topology_context`
  - `get_harmonic_context`
  - `get_verification_snapshot`

## Observed tool patterns by family

- `harmonic_anomaly`: `wls_from_path` -> `get_harmonic_context` -> `run_hse_from_path`
- `measurement_error`: `wls_from_path` -> `correct_measurements_from_path` -> `get_verification_snapshot` -> `wls_from_path`
- `no_error`: `wls_from_path`
- `parameter_error`: `wls_from_path` -> `get_parameter_context` -> `correct_parameters_from_path` -> `get_verification_snapshot` -> `wls_from_path`
- `topology_error`: `wls_from_path` -> `get_topology_context` -> `correct_topology_from_path` -> `get_verification_snapshot` -> `wls_from_path`

## Length observations

Average visible content per trace is now dominated by:
- system prompt: ~3159 chars
- user message: ~2489 chars
- final assistant JSON: ~1331 chars

Visible assistant tool-call payloads are tiny (about 152 chars on average), and visible tool replies are also compact (about 380 chars on average).

User `z_obs` values are mostly rounded to 6 decimals.
Visible tool numeric outputs are mostly rounded to about 4 decimals.

## Important remaining issues

### 1) Topology status text is inconsistent
In all inspected topology traces:
- helper context says breaker `observed_status` is `"open"` and `desired_status` is `true`
- correction tool returns `new_status: true`
- but final `suspect_location.details` says `old_status: "closed"` and `new_status: "open"`

That looks reversed or semantically inconsistent.

### 2) Measurement/parameter verification is still partially hidden-state based
For `measurement_error` and `parameter_error`, the second `wls_from_path` call is visibly the same:
- first WLS: `{"case_path":"case14"}`
- after `get_verification_snapshot`: second WLS is again `{"case_path":"case14"}`

So the model only knows that a *different* snapshot is being used because of hidden runtime state, not because the visible tool argument changes.

This is workable if the runtime really is stateful everywhere, but it makes the traces less self-describing and less reproducible from the visible conversation alone.

### 3) System-prompt output schema still shows `error_family` like a list
The system prompt describes:
```json
"error_family": ["measurement_error", "parameter_error", ...]
```
but the actual final targets correctly use a scalar string like:
```json
"error_family": "parameter_error"
```
This mismatch should be removed.

### 4) Some field bases are mixed
Examples:
- evidence uses `line_row0`
- action uses `line_index`
- some measurement indices are clearly 0-based (`index0`, `lambda_index0`, `indices0`)

If `line_index` is 1-based while `line_row0` is 0-based, the schema should say so explicitly.

### 5) `post_action_success` is too optimistic / ambiguous
In many measurement and parameter traces, post-action residual ratios remain above 1, sometimes far above 1, yet:
```json
"post_action_success": true
```
This likely means “tool executed” or “improved” rather than “resolved”.
That should be renamed or split.

### 6) Harmonic `arguments_hint` wording is misleading
The final harmonic traces use:
```json
{"harmonic_measurements": "provided_in_dialog"}
```
But the full harmonic measurements are not actually visible in the dialog; they live in hidden helper context.
That wording should be adjusted.

## Best next changes

1. Make verification snapshots produce a **visible symbolic alias** for measurement and parameter follow-up too, not just topology.
2. Fix topology `old_status` / `new_status` wording.
3. Make the prompt schema describe `error_family` as a scalar enum, not a list literal.
4. Make indexing bases explicit in every field name or schema description.
5. Replace `post_action_success` with something like:
   - `post_action_executed`
   - `post_action_improved`
   - `post_action_resolved`
6. Change harmonic `arguments_hint` wording to reflect helper-bound context, not “provided_in_dialog”.

## Bottom line

This is now a **good trace design direction** and much closer to what you want for SFT.
The remaining issues are no longer “fundamental data design” issues; they are mostly **contract clarity and semantic consistency** issues.
