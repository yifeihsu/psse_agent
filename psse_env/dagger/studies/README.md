The current study template is `dagger_multiseed_study_v2.json`. Its collection,
frozen-suite evaluation, development evaluation, and recovery-stress evaluation
use the shared 40-action episode horizon. The current BC0 policy keeps JSON
schema version 3 and has policy identity `bc0_closed_loop_hard_gate_v4`.

The v1 study file is preserved byte-for-byte as a record of the earlier
24-action experiment. Its original policy bytes are preserved in
`archive/bc0_evaluation_policy_v3_24_steps.json`. Existing result artifacts are
not relabeled or reevaluated by this migration. A new 40-action result is a
different experimental protocol even when it uses the same physical roots.

Normal manifest loading selects v2. Read-only historical validation must be
explicit:

```python
from psse_env.dagger.study_manifest import ARCHIVED_STUDY_MANIFEST, load_study_manifest

historical = load_study_manifest(ARCHIVED_STUDY_MANIFEST, archive_context=True)
```

`validate_study_manifest`, `validate_study_artifact_binding`, and the two
`canonical_*_evaluation_contract` helpers accept the same explicit archive
context. Historical gate validation must select the archived policy and its
v3 policy identity. New production DAgger collection requires the current
40-action limit. Low-level runners may retain explicit smaller limits for
tests or deliberate experiments; their normal default is 40.
