# Practical GNN scenarios and alignment with SFT

The initial GNN pilot used many smaller disturbances than the SFT pipeline.
The revised training/evaluation population emphasizes meaningful,
measurement-supported disturbances. Physically positive cases below that scope
remain positive in a separate boundary/challenge manifest. They are never
relabeled healthy.

Generation, five-seed retraining, and independent evaluation are complete. The
[V2 results](../research/gnn_screen/results/practical_v2_20260916/README.md)
include a paired frozen-V1 comparison, selection coverage, and a separate
SFT-range HIF challenge. The following rules were fixed before V2 test inference.

Two distinctions govern this revision:

1. Physical or diagnostic importance is different from visibility in the
   permitted balanced measurement vector.
2. Large changes in a known noiseless healthy/fault pair do not prove that a
   classifier facing unknown operating parents and competing causes can identify
   the fault family.

## Verified SFT reference

The live references are `research/hpc/full_pipeline_20260907/pipeline.env`,
`Transmission/generate_measurements.py`, the current-bearing HIF and imbalance
corpora, and `psse_env/providers/scenario_generator.py`.

| Family | SFT generation/admission | Initial GNN pilot | Revised main proposal |
| --- | --- | --- | --- |
| Meter error | Every injected error at least 10 sigma; base corpus 5–15 sigma; single meter and 2–5 same-channel meters | Single biases 3/7/15 sigma | 10–15 sigma, single and multi-meter |
| Parameter | True physical R, X, or RX factors 0.1–0.5 or 2–5, while reported model stays stale | Reported R or X changed by 7/16/30%, physical state unchanged | Physical R/X/RX changed using SFT factors; reported parent model stays fixed |
| Unbalance | Phase fractions Dirichlet(3, 3, 3), total P/Q preserved; acquired VUF/current evidence required | Simple cyclic 8/30/65% redistribution | Symmetric three-phase redistributions with maximum external-bus negative/positive-sequence VUF at least 1% |
| HIF | Rpu 20–200, location 0.25–0.75, ABC; ten diverse scans and branch-current telemetry, relay flag by default | Rpu 120/24/5; single balanced snapshot | Explicit stronger Rpu 5–40 curriculum, with Rpu 20–200 retained as SFT-matched challenge |
| Topology | Dangling line terminals and bus splits in the current full node/breaker pipeline | Connected bus-branch status toggles | Connected physical branch-status changes with a stale reported model; scope remains narrower than full node/breaker SFT |

The parameter direction matters. Multiplying a physical parameter by f with a
fixed reported value is not equivalent to multiplying the reported value by f
while holding physics fixed. The latter would require the reciprocal factor
1/f. The revised physical generator uses the SFT direction directly.

Actual SFT inventory: 620 parameter rows include R: 227, X: 193, RX: 200. Changed
factors span approximately 0.1–5. The unbalance corpus has 220 rows: median
`max(abs(3*f-1))` is 52.4%, its 90th percentile 93.0%, and 117/220 have VUF at least 1%.
The HIF training corpus contains 85 events with 10 scans each, resistance 20.06–198.89 pu,
median 99.59 pu. Thus SFT HIFs are not uniformly stronger than the first pilot;
the SFT diagnostic has more information.

Both HIF implementations use the same normalized 1 kV line-to-line, 100 MVA base,
Zbase 0.01 ohm. The numerical resistance difference is not a missing voltage-base
conversion. These normalized currents must not be interpreted as actual field
protection settings.

## Main-cohort rule

Generate feasible operating parents before fault/noise expansion. Use one
corrected OpenDSS measurement convention and unchanged sensor standard
deviations: 0.001 pu voltage, 0.01 pu injection/terminal power. Split parents before
creating related variants. Keep calibration parents healthy-only.

For each physical candidate, calculate the offline paired separation

\[
D=\left\|R^{-1/2}(z_{\mathrm{fault,clean}}-z_{\mathrm{parent,clean}})\right\|_2.
\]

The **main** cohort requires both its physical/diagnostic criterion and **D>= 5**.
The clean vectors share the same reported model. For physical parameter/status
errors, the true plant changes while the reported model stays fixed; this avoids
treating identical measurements under different model inputs as an identical
statistical hypothesis.

- **Measurement:** every injected bias is 10–15 sigma, including each member of a
  multi-meter error.
- **Parameter:** an active nonzero-R/X line, physical R/X/RX factors in the SFT
  ranges, valid physical solve, and the same paired margin.
- **Unbalance:** actual maximum external-bus `|V2|/|V1|>=0.01`. This 1% is an
  explicit research scope choice. Retain 1–2% and >= 2% strata, all phase orientations,
  actual voltage extrema, and constant-PQ validity checks.
- **HIF:** an injected resistive phase-to-ground fault with charging-corrected
  two-terminal differential-current magnitude at least 6 sigma, using the
  differential noise standard deviation `sqrt(2)*0.001 pu`. Propose Rpu 5–40 in
  three bands; the 5–20 portion deliberately extends beyond the SFT envelope.
- **Topology:** connectivity-changing, connected physical branch-status errors.
  Full bus-split/merge breaker cases require their fixed-meter topology mapping
  and are not claimed covered by this bus-branch cohort.
- **Mixed:** measurement+parameter, measurement+topology, measurement+HIF. Each
  physical component must qualify separately before the meter overlay is added;
  a large meter error cannot promote an otherwise weak HIF into the main cohort.

No candidate is selected by a noisy WLS alarm, a learned score, final-test
performance, or teacher success. A clean fitted-residual energy floor is also
not required: it would discard some physically important examples motivating
this screen. Noiseless WLS energy and local Jacobian projection can be reported
as audit strata, without entering online features or admission decisions.

For a fixed pair of Gaussian means with common covariance, an oracle that
knows that exact pair has 1%-false-positive recall `Phi(D-2.326)`, approximately
99.6% at D=5. This is an optimistic known-parent reference, not a promised GNN
accuracy or proof of multi-family identifiability. In particular, benign
operating changes are nuisance variables for the actual classifier.

## Why physical importance cannot be inferred from a quiet WLS alarm

A fresh development sweep used four new OPF parents, 933 convergent disturbance
conditions, and three explicitly recorded nonconvergences. It did not use the
previous final-test parents or any GNN scores. Representative cases:

| Development case | Physical evidence | Paired D | Clean WLS J | Noisy WLS alarms (256 fresh draws) |
| --- | --- | ---: | ---: | ---: |
| HIF, line 13–14 phase A, Rpu 20 | Fault current 0.05098 pu; normalized-system fault dissipation 1.73 MW | 18.61 | 9.22 | 20/256 (7.81%) |
| Bus 14 unbalance, fractions 0.3/0.1/0.6 | Maximum VUF 3.74% | 10.76 | 9.87 | 18/256 (7.03%) |
| Bus 3 unbalance with unchanged phase-A load fraction | Maximum VUF 2.79% | 1.65 | Near healthy | 3/256 (1.17%) |
| Matched healthy control | No injected error | 0 | Near zero | 5/256 (1.95%) |

The first two qualify for the main cohort while mostly escaping WLS. The third
is materially unbalanced but weak in the permitted measurement channels, so it
belongs in the boundary cohort. Calling it physically trivial would conflate
measurement limitations with physical importance.

Evidence lives under `output/gnn_screen/practical_strength_development_20260916/`
(`sweep.json`, `summary.json`, `noise_check.json`, and reproduction scripts), with
the SFT inventory in `output/gnn_screen/sft_physical_severity_inventory_20260916.json`.
These are development examples, not population detection estimates.

## Engineering context and interpretation limits

Negative-sequence voltage can increase rotating-machine heating, as described
in the [IEC TR 61000-3-13 scope](https://webstore.iec.ch/en/publication/4145).
The 1% VUF cohort threshold is not a claim that a single instantaneous snapshot
violates a universal equipment or network standard. The [DOE motor tip sheet](https://www.energy.gov/sites/prod/files/2014/04/f15/eliminate_voltage_unbalanced_motor_systemts7.pdf)
uses a different NEMA metric based on line-to-line voltage-magnitude deviation;
it must not be equated with sequence VUF.

Low-current HIFs can still pose shock/fire hazards and evade ordinary protection,
according to [EPRI's live downed-conductor work](https://distribution.epri.com/wildfire/public/innovation/live-downed-conductor/).
The omitted low-D cases therefore remain physical positives in a separate
challenge set, and independent phase acquisition remains necessary. The
resistive steady-state model does not simulate arcing or establish field safety.

The legacy SFT unbalance data omit SCADA noise, and its exporter has the earlier
shunt convention. Fresh generation is required; copying those vectors directly
would introduce a source-dependent shortcut. Likewise, SFT recoverability and
parameter-ranking admission are not imported as benchmark filters.

The revised study changes the evaluated population. Its numbers must be
reported as conditional practical-cohort results, with proposal/rejection
counts, phase/asset coverage, and a separate boundary evaluation. A higher
score on this stronger cohort is not an algorithmic improvement on the old
unfiltered distribution.

## Physical resistance update (2026-09-20)

Historical HIF pu values and unmarked `r_hif_ohm` labels in this document describe the normalized 1 kV model. New physical-ohm corpora use local voltage bases and explicit measurement conventions. See [the reconfiguration and fresh results](ieee14_hif_legacy_reconfiguration_20260919.md); old corpora and historical measurements remain unchanged.
