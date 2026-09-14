# IEEE57 training-data readiness

Subsequent P0-P2 implementation and the audited collection/export/replay pilot are
recorded in [the September 13 report](ieee57_audited_training_pilot_20260913.md).
The results below describe the earlier ten-scenario readiness milestone.

## What can be generated now

The basic IEEE14 experiments remain in their original working tree. IEEE57 work
continues in the separate `C:\dw` checkout. The new system can already generate
fresh balanced candidate scenarios for clean operation, single measurement
errors, multiple measurement errors, parameter errors, and combined measurement
and parameter errors.

The physical generator produces solved operating windows, observations, offline
truth, and rejection receipts. The observable expert and existing evaluator can
run those scenarios and record actions, final outcomes, and truth audits. This
establishes candidate generation and teacher execution; it does not make every
teacher action a trustworthy training label.

## Fresh pilot in this checkout

The following commands completed with seed 2026091207:

```sh
python scripts/build_balanced_transfer_scenarios.py --system case57 --output-dir output/ieee57_training_readiness_20260912 --per-family 2 --num-scans 3 --seed 2026091207 --admission-mode physical
python scripts/validate_balanced_transfer.py --scenarios output/ieee57_training_readiness_20260912/scenarios.json --output output/ieee57_training_readiness_20260912/expert_validation.json --progress
```

Generation produced all 10 requested scenarios, two per family, with 10 distinct
physical roots and 10 parent realizations. Physical admission was used, so
undetected faults and expert failures remained in the evaluation.

| Family | Roots | Audited task successes | Strict final resolved outcomes |
|---|---:|---:|---:|
| Clean | 2 | 2 | 2 |
| Single measurement error | 2 | 2 | 0 |
| Multiple measurement errors | 2 | 2 | 0 |
| Parameter error | 2 | 0 | 0 |
| Measurement + parameter | 2 | 1 | 0 |
| Total | 10 | 7 | 2 |

The task-success metric includes independently audited completion after a
controller handoff. It must not be interpreted as seven autonomous resolved
episodes. Both parameter errors were undetected and falsely finalized. One mixed
episode failed and contained an invalid action. There were zero reported false
commits and zero infrastructure errors; healthy components were preserved on all
10 roots. This small pilot is not a population accuracy estimate.

The raw scenarios and complete expert traces remain local under the output path
above. A [compact result and artifact hashes](../research/ieee57/training_readiness_pilot_20260912.json)
are versioned. No model training or final SFT export was performed.

## Remaining blockers and qualifications

1. **Trustworthy supervision.** Do not export every expert step as a correct
   target. Apply the existing process, target, and healthy-component preservation
   audits; quarantine failed or unsupported labels. Preserve failures in the
   development/evaluation population. A curated recoverable training subset must
   be labelled as such. Undetectable faults also require an explicit evaluation
   policy: hidden truth does not supply missing observable evidence to the agent.

2. **Logical topology integration.** The IEEE57 logical-CB simulator and complete
   hypothesis tester work, but the standard DAgger environment/protocol/private
   audit does not yet support the raw-section/coupler contract.
   `LogicalTopologyProviders.env_kwargs()` deliberately raises. A logical-aware
   adapter must carry the 530-channel evidence, stable device identities,
   single/multiple status corrections, certificates, and final-status audits
   through collection and export. See [provider boundaries](../logical_topology/provider.py#L84).

3. **Mixed topology and analog/parameter errors.** The logical provider's
   measurement and parameter correction routes are unimplemented. The current
   correction certificate also requires an absolutely plausible full fit, so an
   intermediate topology repair leaving another anomaly needs separate semantics
   and validation. More physical simulations alone will not complete this path.

4. **HIF/unbalance agent integration.** The three-phase testbed can generate and
   diagnose physical disturbances, but the existing case57 scenario registry and
   phase-data loader do not route them into the standard agent training workflow.
   Acquisition policy, phase observation schemas, tool responses, and offline
   targets must be connected. Do not assume balanced WLS always triggers the phase
   measurements: previous development testing found many phase-diagnosable faults
   without a WLS alarm. The HIF model is a fundamental-frequency resistive
   surrogate, not a validated nonlinear arcing/harmonic model.

5. **Independent, adequately covered splits.** Existing outputs are development
   evidence. Assign independent physical parents to train, validation, and test
   before creating noise/overlay derivatives. Changing only the topology seed
   does not create a new physical world: the parent identity depends on the
   operating point, layout, and true statuses. Existing structural holdout leaves
   several topology families absent from training. New operating worlds and
   explicit family/device coverage checks are required; moving existing rows
   between labels is insufficient.

6. **Complete collection and export verification.** Generic rollout collection
   and canonical chat-SFT export exist, but this pilot did not verify a complete
   IEEE57 collect-to-audit-to-export training release. This is the next check for
   the supported balanced families. A larger run should include post-correction
   states, failed-policy recovery states, and audited stopping behavior.

Missing NLM ranking is **not a fundamental blocker** to starting topology
supervision with the current exhaustive hypothesis tester. It becomes a
requirement for experiments specifically claiming an NLM-guided diagnostic
policy. The [standalone feasibility study](ieee57_topology_method_review_20260912.md)
does not yet provide that production integration.

## Recommended sequence

Start with the five balanced families: generate independent source windows,
collect expert trajectories, run the strict supervision audits, and verify a
small canonical SFT export before scaling. Keep a separate unfiltered evaluation
population. Then connect the logical topology contract to that same audited
pipeline, followed by mixed errors and three-phase acquisition/diagnosis.

The numerical source/testbed suite passed **403 tests and 45 subtests in 91.21
seconds** before this branch was committed. The portable coupler-NLM probe also
passed its 11 feasibility checks. These software checks do not certify teacher
quality or training-release readiness.

Install numerical/test dependencies using
`python -m pip install -r psse_env/requirements-ieee57.txt`. Large generated
datasets, models, local receipts, scratch files, and Git transfer bundles are
excluded from Git. Source code, tests, model specifications, scripts, and compact
review results are versioned; no files or HPC jobs in the original working tree
were modified by this commit preparation.
