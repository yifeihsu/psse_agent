# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 50 | 100% | 12.4 | 3.36 | 5 | 50 / 55 / 0 / 0 | 0 / 13 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.7 | 2.06 | 24 | 48 / 0 / 0 / 1 | 4 / 0 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 13 | 17% | 18.9 | 5.45 | 35 | 13 / 43 / 59 / 2 | 50 / 163 | 71 | 4 | 0 | 0 (0) |
| meter_rx2 | 150 | 88 | 59% | 18.5 | 4.79 | 166 | 117 / 292 / 37 / 23 | 25 / 73 | 148 | 2 | 0 | 0 (0) |
| rx2_adjacent | 150 | 61 | 41% | 17.6 | 4.46 | 262 | 101 / 202 / 15 / 46 | 63 / 92 | 150 | 0 | 0 | 0 (0) |
| rx2_distant | 150 | 115 | 77% | 12.4 | 3.25 | 104 | 132 / 140 / 7 / 15 | 16 / 27 | 149 | 1 | 0 | 0 (0) |
| all | 625 | 375 | 60% | 15.5 | 4.09 | 596 | | | | | | 0 (0) |
