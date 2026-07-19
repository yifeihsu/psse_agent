# Multi-Scan HIF Snapshot Benchmark

Date: 2026-07-13

## Experiment

Eight matched synthetic IEEE-14 HIF events were generated. Every event used
the same branch, alpha, phase, and HIF resistance across its scan window. NLM
selected the correct branch in all eight events.

The comparison fixed phase to the known synthetic phase so the experiment
isolated alpha and resistance estimation. The common comparison search used a
7 x 9 alpha/resistance grid, two refinement seeds, 20 local evaluations, and
unit measurement-noise scale. A 13 x 15 grid was also run on three matched
events to check whether the result was a coarse-grid artifact.

## Snapshot Curve

The diverse-scan rows use information-greedy selection from 20 candidate
snapshots. Errors are over eight matched events.

| Condition | Median alpha error | P90 alpha error | Alpha <= 0.10 | Median R error | P90 R error | Median diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 scan | 0.472 | 0.586 | 25.0% | 18.4% | 71.8% | 0.0000 |
| 3 diverse | 0.400 | 0.500 | 25.0% | 17.7% | 38.2% | 0.0038 |
| 5 diverse | 0.311 | 0.567 | 25.0% | 8.6% | 18.3% | 0.0039 |
| 10 identical-noise | 0.213 | 0.519 | 0.0% | 10.6% | 17.8% | 0.0000 |
| 10 diverse | 0.193 | 0.546 | 37.5% | 11.8% | 22.4% | 0.0055 |
| 15 diverse | 0.243 | 0.455 | 0.0% | 7.1% | 19.3% | 0.0110 |
| 20 diverse | 0.205 | 0.400 | 0.0% | 4.6% | 22.5% | 0.0095 |

Relative to one scan, alpha error improved in 5/8 identical-noise windows,
5/8 ten-scan diverse windows, and 6/8 twenty-scan diverse windows. Resistance
error improved in the same counts. These paired improvements are directional,
but eight events are not enough to claim statistical significance.

All evaluated point estimates were marked ambiguous, false precision was zero,
and the near-best alpha interval covered truth in the four primary conditions.
The multi-scan observability status was `noise_averaging_only` whenever the
finite-difference diagnostic completed.

## Denser-Grid Check

The 13 x 15 check on the first three matched events reproduced the conclusion:

| Condition | Median alpha error | Median R error |
| --- | ---: | ---: |
| 1 scan | 0.473 | 68.5% |
| 10 identical-noise | 0.444 | 17.0% |
| 10 diverse | 0.065 | 9.1% |
| 20 diverse | 0.217 | 8.5% |

The diverse ten-scan result helped two events substantially but did not resolve
the weak third event. Twenty scans improved resistance but did not improve
alpha monotonically.

## Stronger Diversity Check

A matched ten-scan condition increased bounded bus-load, dispatch, and voltage
variation. Median sensitivity diversity rose from 0.0055 to 0.0229. Its median
alpha error was 0.194, P90 alpha error was 0.353, and median resistance error
was 11.3%. Stronger variation therefore improved the alpha-error tail, but it
did not materially improve median alpha and introduced finite-difference solve
failures in two events.

## Conclusion

More snapshots provide a clear noise-averaging benefit for HIF resistance. The
median resistance error fell from 18.4% with one scan to 4.6% with twenty
diverse scans.

Alpha localization benefits are real for some events but are not monotonic or
reliable with the current measurements. Default scan sensitivity directions
remain nearly collinear, so adding snapshots mostly adds information magnitude
rather than new information geometry.

For the current synthetic pipeline, collect up to 20 candidate snapshots and
use information-greedy selection of roughly 10 for the joint alpha/resistance
fit. Keep the ambiguity and observability gates enabled. Improving alpha beyond
this point likely requires more informative measurements, such as synchronized
terminal voltage/current phasors or per-phase line-flow measurements, rather
than simply extending the window.

## Numerical Hardening

This benchmark exposed isolated OpenDSS failures at finite-difference
perturbations. Shared-resistance diagnostics now retain successful scan
information, report `diagnostic_partial`, and force
`parameter_identifiable=false`. Information-greedy pilot selection also
excludes scans whose pilot sensitivities cannot be computed.
