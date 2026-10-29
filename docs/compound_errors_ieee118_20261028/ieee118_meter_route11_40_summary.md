# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 50 | 100% | 11.2 | 3.04 | 0 | 50 / 51 / 0 / 0 | 0 / 1 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 45 | 90% | 8.0 | 2.16 | 28 | 46 / 0 / 0 / 3 | 3 / 6 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 11 | 15% | 21.9 | 6.37 | 59 | 15 / 35 / 68 / 4 | 55 / 222 | 72 | 3 | 0 | 0 (0) |
| meter_rx2 | 150 | 130 | 87% | 17.1 | 4.26 | 202 | 136 / 287 / 3 / 10 | 16 / 37 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 95 | 63% | 15.7 | 4.10 | 187 | 114 / 171 / 5 / 35 | 46 / 93 | 150 | 0 | 0 | 0 (0) |
| rx2_distant | 150 | 130 | 87% | 13.0 | 3.27 | 183 | 131 / 146 / 2 / 13 | 21 / 28 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 461 | 74% | 15.2 | 3.97 | 659 | | | | | | 0 (0) |
