# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 49 | 98% | 21.6 | 5.70 | 34 | 49 / 55 / 0 / 0 | 0 / 131 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.7 | 2.06 | 24 | 48 / 0 / 0 / 1 | 4 / 0 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 13 | 17% | 19.1 | 5.48 | 35 | 13 / 42 / 62 / 2 | 46 / 171 | 75 | 0 | 0 | 0 (0) |
| meter_rx2 | 150 | 88 | 59% | 17.7 | 4.60 | 149 | 118 / 265 / 33 / 23 | 26 / 75 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 86 | 57% | 15.4 | 3.97 | 210 | 108 / 139 / 14 / 48 | 54 / 83 | 150 | 0 | 0 | 0 (0) |
| rx2_distant | 150 | 115 | 77% | 12.4 | 3.25 | 104 | 132 / 140 / 7 / 15 | 16 / 28 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 399 | 64% | 15.5 | 4.12 | 556 | | | | | | 0 (0) |
