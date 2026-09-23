# Balanced SCADA discovery protocol

The current research default is `evidence_profile="scada_only"`. The agent starts
with no supplied HIF, unbalance or harmonic diagnosis and must run balanced WLS
before obtaining correction contexts or closing an episode. The action budget
remains 40. This implements the requirement that decisions use the balanced
SCADA observations, with no additional sensor signals supplied to the agent.

The previous flagged HIF configuration was incompatible with that requirement.
Filtering a corpus for a WLS alarm is an offline admission decision; it does not
justify passing the injected fault family into an episode. Merely selecting the
older `signature_mode="discovered"` was insufficient: that route could still
request three-phase or harmonic telemetry after WLS.

## Evidence available during execution

The strict profile retains the network case, original balanced SCADA readings,
declared measurement noise/covariance and measurement conventions. If observed
SCADA history is supplied, its readings and noise information are available;
trusted simulator operating points and initial states are not. WLS results and
measurement, parameter and topology contexts computed from these inputs remain
available. Optional learned screening may use the same SCADA/WLS inputs, but it
does not grant access to another sensor channel.

The controller and numerical-provider boundaries remove preseeded fault flags,
hidden family/action hints, phase voltages, phase branch currents, harmonic
measurements, cached NLM diagnoses, HIF fits and auxiliary acquisition metadata.
Auxiliary tool calls are rejected before numerical dispatch. Learned-policy tool
schemas are restricted to the same capabilities. Truth stays in the offline
audit; it cannot choose an action or accept a candidate at runtime.

The default HIF signature mode is `discovered`. Explicit flagged HIF generation
is rejected under `scada_only`. Historical reproduction requires an explicit
`auxiliary_diagnostics` profile, which may use the old flagged or discovered
signature settings. Do not compare these profiles without naming the different
available measurements.

## Detection is not HIF identification

Balanced WLS detects a mismatch using the chi-square statistic and normalized
residual alarm. Its alarm alone does not uniquely establish HIF, faulted phase,
location or resistance. The existing three-phase HIF estimator and the earlier
HIF-conditioned meter-repair path require auxiliary measurements and are disabled
in this profile. There is no substitution of hidden HIF truth for their output,
and no subtraction of a simulator-derived HIF contribution from SCADA readings.

The expert can investigate and repair errors supported by balanced evidence. If
the supported investigations cannot resolve a model discrepancy, it can request
a generic operator handoff. Such a handoff is not HIF identification, physical
fault removal or successful mixed-fault repair. The offline truth audit remains
necessary, including for a statistically quiet result obtained after correction.

## Training and evaluation compatibility

New collection, factories and the current full-pipeline launch configuration use
the strict profile. Profile provenance is recorded with collected/exported data.
The research runner checks both new and replayed training inputs: old rows with
missing provenance cannot silently become strict SCADA labels. The pipeline also
checks profile compatibility before reusing earlier D0/BC0 stages. A new strict
training run therefore needs compatible fresh collection; changing a flag does
not convert old auxiliary-assisted trajectories or checkpoint weights.

Frozen evaluation inputs are sanitized at execution without redrawing their
noise or removing roots. Historical correction-setup interventions that begin
with a context request now receive an explicit initial WLS action in a strict
environment. It is recorded as setup in the trace and charged to the same
40-action budget; it is not supplied as a precomputed diagnostic hint.
Existing auxiliary-assisted reports remain historical.
In particular, the revised expert's 158/160 and R2's 157/160 final-state audit
scores are not SCADA-only results. The latter also included four nonterminal
episodes; it was not a 157/160 terminal task-completion result.

No training, corpus regeneration or HPC evaluation was launched as part of this
protocol change. A new full evaluation is needed to measure identification and
repair performance under this evidence restriction.

## Verification

The provider, controller and expert tests check that changing auxiliary streams
and hidden labels while preserving the case, SCADA readings and noise model
leaves state hashes, WLS results and selected actions unchanged. Tests also check
blocked auxiliary calls, current-WLS preconditions and explicit historical
compatibility.

An initial-WLS smoke used all eight unchanged frozen HIF-plus-meter roots from
`output/hif_continuation_fix_20260922/frozen_mixed_scenarios.json`:

- All eight started with `run_wls`, then selected `get_measurement_context`.
- All eight triggered both the chi-square alarm (alpha 0.01) and normalized
  residual alarm (threshold 4).
- Case readings and sigma values were unchanged. Perturbing all auxiliary and
  hidden diagnosis fields changed neither WLS nor the selected actions.

This checks alarm visibility and routing only; it is not a completed-episode
repair evaluation or a clean-control calibration study. The script and detailed
statistics are saved under `output/scada_only_protocol_20260922/`.
