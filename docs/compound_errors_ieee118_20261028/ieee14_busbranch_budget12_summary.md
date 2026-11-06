# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 14 | 28% | 9.8 | 2.74 | 24 | 14 / 39 / 0 / 0 | 0 / 34 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.5 | 2.02 | 23 | 48 / 0 / 0 / 1 | 2 / 0 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 11 | 15% | 10.6 | 3.20 | 25 | 12 / 34 / 50 / 0 | 7 / 68 | 74 | 1 | 0 | 0 (0) |
| meter_rx2 | 150 | 0 | 0% | 11.6 | 3.05 | 114 | 8 / 246 / 18 / 8 | 8 / 21 | 139 | 11 | 0 | 0 (0) |
| rx2_adjacent | 150 | 44 | 29% | 11.4 | 3.02 | 177 | 56 / 170 / 3 / 36 | 30 / 12 | 104 | 46 | 0 | 0 (0) |
| rx2_distant | 150 | 115 | 77% | 11.4 | 2.99 | 104 | 132 / 140 / 6 / 12 | 5 / 5 | 149 | 1 | 0 | 0 (0) |
| all | 625 | 232 | 37% | 10.9 | 2.94 | 467 | | | | | | 0 (0) |
