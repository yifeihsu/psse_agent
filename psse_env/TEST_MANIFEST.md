# DAgger test manifest

The archive contains the main scaffold suite plus focused verifier, production
export, and Gemma SFT gates. These archive-contained suites execute 188 tests.
Full-workspace discovery executed 274 tests; the additional tests belong to
other repository modules and are intentionally outside this archive.

| Group | Test class | Contract |
|---|---|---|
| state-store tests | `StateStoreTests` | isolation, hashes, lineage, provenance |
| transaction tests | `TransactionTests` | central process gate and candidate lifecycle |
| process-validity tests | `ProcessValidityTests` | preconditions, references, repair actions |
| candidate-quality tests | `CandidateQualityTests` | final/partial/reject/inconclusive labels |
| expert-policy tests | `ExpertPolicyTests` | recovery and domain proposal routing |
| DAgger-collector tests | `DaggerCollectorTests` | failure collection, updated history, best checkpoint |
| dataset-conversion tests | `DatasetConversionTests` | leakage gate, JSONL, native tool-call SFT |
| replay/evaluation tests | `ReplayAndEvaluationTests` | balanced replay and grouped splits |
| counterfactual tests | `CounterfactualTests` | isolated recovery branches |
| process-verifier tests | `ProcessVerifierTests` | deterministic rules and safety metrics |
| verifier hardening tests | `Verifier*HardeningTests` | episode provenance, schema, continuity, leakage |
| AggreVaTe-lite tests | `AggreVaTeLiteTests` | isolated top-L ranking and branch costs |
| SFT export tests | `psse_env.dagger.test_sft_export` | native schemas, dict arguments, aliases, provenance, bounded history, conflict audit |
| production-mode tests | `psse_env.test_production_mode` | provider declarations, evidence sufficiency, realizable labels, target-aware classes |
| exact-gate offline tests | `psse_env.sft.tests.test_gates` | schema conformance, masks, truncation, grouped production rows, smoke logic |

Preferred archive-contained gate after installing `psse_env/requirements-dev.txt`:

```bash
pytest -q \
  test_psse_dagger_scaffold.py \
  psse_env/verifier/test_hardening.py \
  psse_env/dagger/test_sft_export.py \
  psse_env/test_production_mode.py \
  psse_env/sft/tests/test_gates.py
```

Discovery gate from the full repository workspace:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q
```

From a clean extraction containing only this archive, the same discovery command
runs the 188 archive-contained tests. The broader 274 count is expected only in
the original workspace.

The live model checks are separate from both offline counts. See
`SFT_PILOT_VALIDATION.md` and `sft/LIVE_GATE_RESULTS.md` for the exact pinned
31B processor gate and E2B QLoRA optimizer/generation smoke results.
