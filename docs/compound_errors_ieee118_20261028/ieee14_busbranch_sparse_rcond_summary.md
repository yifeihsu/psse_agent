# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 49 | 98% | 21.6 | 5.70 | 34 | 49 / 55 / 0 / 0 | 0 / 131 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.7 | 2.06 | 24 | 48 / 0 / 0 / 1 | 4 / 0 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 13 | 17% | 19.1 | 5.48 | 35 | 13 / 42 / 62 / 2 | 46 / 171 | 75 | 0 | 0 | 0 (0) |
| meter_rx2 | 150 | 88 | 59% | 18.6 | 4.81 | 165 | 117 / 292 / 39 / 24 | 22 / 78 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 61 | 41% | 17.6 | 4.47 | 264 | 101 / 202 / 17 / 48 | 62 / 90 | 150 | 0 | 0 | 0 (0) |
| rx2_distant | 150 | 115 | 77% | 12.4 | 3.25 | 104 | 132 / 140 / 7 / 15 | 16 / 28 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 374 | 60% | 16.3 | 4.29 | 626 | | | | | | 0 (0) |
