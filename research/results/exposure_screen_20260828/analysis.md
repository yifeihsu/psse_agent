# Frozen A-D occupancy/exposure screen

Data-quality gate: **passed** (65 aligned physical roots: 61 faulted + 4 no-error).
Scenario SHA-256: `16fff49570a77b66001adcd95a39277e32395fec868d2fd8f618e61d863530af`.

Provenance limitation: Historical source evaluations lack suite hashes and per-episode identities; their replay-complete audits are aligned by immutable ordered replay. Paired historical comparisons inherit this weaker lineage and must not be described as identity-bound source evaluations.

| Policy | Exact | Stable exact | Event precision | Unique precision | Recall | Faulted FI | Partial | Aborts | Loops | Mean excess steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 35/61 | 35/61 | 87/94 (92.6%) | 87/94 (92.6%) | 87/150 (58.0%) | 6 | 18 | 0 | 0 | 0.78 |
| B | 31/61 | 31/61 | 80/95 (84.2%) | 80/93 (86.0%) | 80/150 (53.3%) | 12 | 17 | 0 | 2 | 0.68 |
| C | 43/61 | 43/61 | 98/100 (98.0%) | 98/99 (99.0%) | 98/150 (65.3%) | 0 | 17 | 0 | 1 | 0.09 |
| D | 25/61 | 25/61 | 64/78 (82.1%) | 64/76 (84.2%) | 64/150 (42.7%) | 11 | 16 | 0 | 9 | 1.88 |

## Input provenance

| Policy | Audit SHA-256 | Source evaluation SHA-256 | Binding |
| --- | --- | --- | --- |
| A | `a93bd79aeb265edd8a7a487216b13b5a22ede93f38820d8ac01fab2cd67390fe` | `475917fb219a89cb82ad0279e838833db80526a91f7a2415f93a3a536981062d` | `sha256_and_per_episode_identity` |
| B | `b250d031731be7906cf0df7b7969a38ad252f4e36c2a53c09d400b3c6cbe4bd5` | `73a49e0056c3ee7eb6d8f36908e617d3db0597977d865608fcfa4f3c56b7feae` | `sha256_and_per_episode_identity` |
| C | `8c44ab5ade8e8e679f783473b385c5aa8146e8a2693768488278d746d2404536` | `04755b67434064346eb1bc22b20bc3378b910f252591f9d758ea11cc32f83815` | `sha256_and_per_episode_identity` |
| D | `38cf901dfa6e1024b000ea10335c66b8c28d4ed7186642ef0afb5bc6b533c06a` | `330e1c97bf7dfb9965e65abcb18b28a8ffaaea285ddd6887cbfbe46299d1f774` | `sha256_and_per_episode_identity` |
| historical_e2b_selective | `ea89434d24bdf4d9df36b71889d0c9d5e25bf301b805bfd58688b2680aae5ea8` | `d34bf62f1aee3b49d5c2f71331d1547312ba0f87882161564e00415d918be992` | `ordered_replay_without_source_suite_hash` |
| historical_e2b_full | `ea89434d24bdf4d9df36b71889d0c9d5e25bf301b805bfd58688b2680aae5ea8` | `9ca26350f29a9fefc1b087dafdec2aac11b3d08a8e030ce67e303fa7e2da28c7` | `ordered_replay_without_source_suite_hash` |
| historical_12b_selective | `ea89434d24bdf4d9df36b71889d0c9d5e25bf301b805bfd58688b2680aae5ea8` | `94ac8b35089025e84a2ea5f329d6897720f983bc33035eb0f797ab7ae2b59395` | `ordered_replay_without_source_suite_hash` |
| historical_12b_full | `ea89434d24bdf4d9df36b71889d0c9d5e25bf301b805bfd58688b2680aae5ea8` | `594f54567bde7a7bb308d520a0766d3c02b48222f1fb33adb5f9798a8efcbbb9` | `ordered_replay_without_source_suite_hash` |

## No-error preservation

| Policy | Clean preserved | No-error false interventions |
| --- | ---: | ---: |
| A | 4/4 | 0 |
| B | 4/4 | 0 |
| C | 4/4 | 0 |
| D | 4/4 | 0 |

## Six-way trajectory outcomes (all 65 episodes)

| Policy | Exact | Partial | False intervention | No progress | Abort | Loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 39 | 18 | 6 | 2 | 0 | 0 |
| B | 35 | 16 | 12 | 0 | 0 | 2 |
| C | 47 | 17 | 0 | 0 | 0 | 1 |
| D | 29 | 16 | 11 | 0 | 0 | 9 |

## Family-level final exact recovery (faulted episodes)

| Policy | Family | Exact | Stable exact |
| --- | --- | ---: | ---: |
| A | measurement+parameter | 18/19 | 18/19 |
| A | measurement+topology | 11/16 | 11/16 |
| A | multi-measurement | 2/20 | 2/20 |
| A | parameter | 3/3 | 3/3 |
| A | topology | 1/3 | 1/3 |
| B | measurement+parameter | 19/19 | 19/19 |
| B | measurement+topology | 7/16 | 7/16 |
| B | multi-measurement | 2/20 | 2/20 |
| B | parameter | 3/3 | 3/3 |
| B | topology | 0/3 | 0/3 |
| C | measurement+parameter | 19/19 | 19/19 |
| C | measurement+topology | 16/16 | 16/16 |
| C | multi-measurement | 2/20 | 2/20 |
| C | parameter | 3/3 | 3/3 |
| C | topology | 3/3 | 3/3 |
| D | measurement+parameter | 12/19 | 12/19 |
| D | measurement+topology | 8/16 | 8/16 |
| D | multi-measurement | 2/20 | 2/20 |
| D | parameter | 3/3 | 3/3 |
| D | topology | 0/3 | 0/3 |

## Paired exact McNemar comparisons

All exact McNemar p-values are two-sided, unadjusted, and exploratory; 21 correlated tests are reported without multiplicity correction.

| Pair | Metric | Both | Policy 1 only | Policy 2 only | Neither | Exact p (unadjusted) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C_vs_historical_e2b_selective | final_exact_recovery | 32 | 11 | 0 | 18 | 0.0010 |
| C_vs_historical_e2b_selective | stable_exact_recovery | 31 | 12 | 0 | 18 | 0.0005 |
| C_vs_historical_e2b_selective | final_false_intervention | 0 | 0 | 0 | 61 | 1.0000 |
| C_vs_A | final_exact_recovery | 35 | 8 | 0 | 18 | 0.0078 |
| C_vs_A | stable_exact_recovery | 35 | 8 | 0 | 18 | 0.0078 |
| C_vs_A | final_false_intervention | 0 | 0 | 6 | 55 | 0.0312 |
| C_vs_historical_12b_selective | final_exact_recovery | 43 | 0 | 0 | 18 | 1.0000 |
| C_vs_historical_12b_selective | stable_exact_recovery | 39 | 4 | 0 | 18 | 0.1250 |
| C_vs_historical_12b_selective | final_false_intervention | 0 | 0 | 0 | 61 | 1.0000 |
| A_vs_historical_e2b_selective | final_exact_recovery | 25 | 10 | 7 | 19 | 0.6291 |
| A_vs_historical_e2b_selective | stable_exact_recovery | 24 | 11 | 7 | 19 | 0.4807 |
| A_vs_historical_e2b_selective | final_false_intervention | 0 | 6 | 0 | 55 | 0.0312 |
| B_vs_historical_12b_selective | final_exact_recovery | 31 | 0 | 12 | 18 | 0.0005 |
| B_vs_historical_12b_selective | stable_exact_recovery | 28 | 3 | 11 | 19 | 0.0574 |
| B_vs_historical_12b_selective | final_false_intervention | 0 | 12 | 0 | 49 | 0.0005 |
| B_vs_historical_12b_full | final_exact_recovery | 31 | 0 | 6 | 24 | 0.0312 |
| B_vs_historical_12b_full | stable_exact_recovery | 31 | 0 | 5 | 25 | 0.0625 |
| B_vs_historical_12b_full | final_false_intervention | 11 | 1 | 0 | 49 | 1.0000 |
| D_vs_historical_12b_selective | final_exact_recovery | 25 | 0 | 18 | 18 | 0.0000 |
| D_vs_historical_12b_selective | stable_exact_recovery | 23 | 2 | 16 | 20 | 0.0013 |
| D_vs_historical_12b_selective | final_false_intervention | 0 | 11 | 0 | 50 | 0.0010 |

## C versus historical low-exposure E2B full occupancy

Descriptive, non-additive episode transitions; categories can overlap and do not identify a causal mechanism.

- Exact recovery: 23/61 -> 43/61.
- Rescued roots: 20; regressed roots: 0; net gain: 20.
- Rescued by family: `{"measurement+parameter": 4, "measurement+topology": 11, "multi-measurement": 2, "topology": 3}`.
- Rescued from abort: 5; loop: 3; false intervention: 10.
- Rescued with more committed target events: 15; summed event delta: 16.

## Selection gate

Strictly eligible A-D arms: `['C']`.
Provisional leader under the stated hard rule: `C`.

This is research evidence, not release evidence.
