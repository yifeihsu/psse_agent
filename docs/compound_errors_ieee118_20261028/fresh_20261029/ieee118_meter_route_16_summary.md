# Expert baseline_compare_meter_route under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 50 | 100% | 11.1 | 3.02 | 0 | 50 / 51 / 0 / 0 | 0 / 0 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 44 | 88% | 7.9 | 2.16 | 27 | 45 / 0 / 0 / 3 | 4 / 6 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 5 | 7% | 14.0 | 4.28 | 46 | 7 / 26 / 58 / 2 | 21 / 134 | 74 | 1 | 0 | 0 (0) |
| meter_rx2 | 150 | 114 | 76% | 15.7 | 4.07 | 187 | 119 / 275 / 13 / 9 | 11 / 33 | 145 | 5 | 0 | 0 (0) |
| rx2_adjacent | 150 | 99 | 66% | 12.8 | 3.37 | 158 | 112 / 159 / 5 / 27 | 17 / 39 | 145 | 5 | 0 | 0 (0) |
| rx2_distant | 150 | 125 | 83% | 12.4 | 3.11 | 185 | 130 / 148 / 1 / 8 | 12 / 18 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 437 | 70% | 13.0 | 3.46 | 603 | | | | | | 0 (0) |
