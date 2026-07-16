# DAgger scaffold baseline

The pre-revision scaffold is frozen in the existing repository checkpoint:

```text
dagger_scaffold_reviewed_20260708_012357.tgz
sha256 b4c1fe595adf21d30c77f518b55c10993789fd9368e61f6100c34dfba89a8246
```

That archive contains the original `psse_env/` package and the original
10-test `test_psse_dagger_scaffold.py`. It is used instead of a Git tag because
the scaffold was untracked in the dirty working tree and therefore was not
part of `HEAD`.

The archived suite was verified before revision with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest -q test_psse_dagger_scaffold.py
# Ran 10 tests ... OK
```

The deterministic post-checkpoint rollout and its two training exports live in
`psse_env/examples/`. Regenerate them twice with the same seed and compare
their SHA-256 hashes to exercise the reproducibility gate.

The archive-contained revision is guarded by 188 focused tests across the
transactional, verifier, production-export, and SFT-gate suites. Full-workspace
discovery also passed 274 tests, including repository tests that are not packed
in this archive (see `TEST_MANIFEST.md`).

```bash
python -m psse_env.examples.generate_baseline
```

Expected fixture hashes:

```text
0a8bfe99e58fd9af347873dcc88563f5e3b4f6b79d38142258ad9cc8e50e8dd1  sample_rollout.jsonl
5d21b542da1c3ba62c5975e73ebb424a4fdc744581716326dc09a0e05c7112bc  dagger_example.jsonl
20f47ff1c90b634d5b91a142d410b5f8849311cccbe14a9f10bd57c1052a9e5b  chat_sft_example.jsonl
```
