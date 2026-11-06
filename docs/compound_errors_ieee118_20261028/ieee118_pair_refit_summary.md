# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 45 | 90% | 21.6 | 5.78 | 31 | 45 / 51 / 0 / 0 | 0 / 143 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 45 | 90% | 8.0 | 2.16 | 28 | 46 / 0 / 0 / 3 | 3 / 6 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 11 | 15% | 22.5 | 6.49 | 63 | 15 / 34 / 70 / 5 | 49 / 240 | 75 | 0 | 0 | 0 (0) |
| meter_rx2 | 150 | 130 | 87% | 17.2 | 4.29 | 204 | 136 / 288 / 4 / 10 | 18 / 37 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 110 | 73% | 14.7 | 3.89 | 167 | 117 / 141 / 6 / 35 | 40 / 94 | 150 | 0 | 0 | 0 (0) |
| rx2_distant | 150 | 130 | 87% | 13.0 | 3.27 | 183 | 131 / 146 / 2 / 13 | 21 / 28 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 471 | 75% | 15.8 | 4.16 | 676 | | | | | | 0 (0) |
