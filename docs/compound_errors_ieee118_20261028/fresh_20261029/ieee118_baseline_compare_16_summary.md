# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 18 | 36% | 12.3 | 3.60 | 19 | 18 / 34 / 0 / 0 | 0 / 78 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 44 | 88% | 7.9 | 2.16 | 27 | 45 / 0 / 0 / 3 | 4 / 6 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 5 | 7% | 14.0 | 4.28 | 48 | 7 / 26 / 58 / 2 | 20 / 135 | 74 | 1 | 0 | 0 (0) |
| meter_rx2 | 150 | 114 | 76% | 15.7 | 4.07 | 187 | 119 / 275 / 13 / 9 | 11 / 33 | 145 | 5 | 0 | 0 (0) |
| rx2_adjacent | 150 | 99 | 66% | 12.8 | 3.37 | 158 | 112 / 159 / 5 / 27 | 17 / 39 | 145 | 5 | 0 | 0 (0) |
| rx2_distant | 150 | 125 | 83% | 12.4 | 3.11 | 185 | 130 / 148 / 1 / 8 | 12 / 18 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 405 | 65% | 13.1 | 3.51 | 624 | | | | | | 0 (0) |
