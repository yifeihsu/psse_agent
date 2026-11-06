# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 18 | 36% | 12.5 | 3.60 | 24 | 18 / 39 / 0 / 0 | 0 / 73 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.6 | 2.04 | 23 | 48 / 0 / 0 / 1 | 3 / 0 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 11 | 15% | 12.9 | 3.88 | 28 | 12 / 34 / 51 / 1 | 16 / 103 | 74 | 1 | 0 | 0 (0) |
| meter_rx2 | 150 | 77 | 51% | 15.2 | 3.97 | 141 | 90 / 275 / 21 / 13 | 11 / 38 | 138 | 12 | 0 | 0 (0) |
| rx2_adjacent | 150 | 59 | 39% | 13.5 | 3.43 | 230 | 78 / 183 / 4 / 41 | 36 / 27 | 139 | 11 | 0 | 0 (0) |
| rx2_distant | 150 | 115 | 77% | 11.9 | 3.13 | 104 | 132 / 140 / 6 / 15 | 8 / 20 | 149 | 1 | 0 | 0 (0) |
| all | 625 | 328 | 52% | 12.9 | 3.44 | 550 | | | | | | 0 (0) |
