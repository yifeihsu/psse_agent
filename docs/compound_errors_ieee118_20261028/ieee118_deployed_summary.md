# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 45 | 90% | 21.6 | 5.78 | 31 | 45 / 51 / 0 / 0 | 0 / 143 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 45 | 90% | 8.0 | 2.16 | 28 | 46 / 0 / 0 / 3 | 3 / 6 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 11 | 15% | 22.6 | 6.52 | 64 | 15 / 34 / 70 / 5 | 50 / 241 | 75 | 0 | 0 | 0 (0) |
| meter_rx2 | 150 | 130 | 87% | 17.1 | 4.26 | 202 | 136 / 287 / 3 / 10 | 16 / 37 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 95 | 63% | 15.7 | 4.11 | 187 | 114 / 171 / 6 / 35 | 45 / 95 | 150 | 0 | 0 | 0 (0) |
| rx2_distant | 150 | 130 | 87% | 13.0 | 3.27 | 183 | 131 / 146 / 2 / 13 | 21 / 28 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 456 | 73% | 16.1 | 4.21 | 695 | | | | | | 0 (0) |
