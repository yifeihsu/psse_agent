# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 17 | 34% | 9.4 | 2.66 | 21 | 17 / 33 / 0 / 0 | 0 / 33 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 45 | 90% | 7.7 | 2.10 | 28 | 46 / 0 / 0 / 3 | 2 / 4 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 4 | 5% | 10.7 | 3.31 | 40 | 6 / 16 / 58 / 3 | 8 / 92 | 74 | 1 | 0 | 0 (0) |
| meter_rx2 | 150 | 0 | 0% | 11.7 | 3.01 | 169 | 0 / 278 / 2 / 6 | 9 / 8 | 100 | 50 | 0 | 0 (0) |
| rx2_adjacent | 150 | 94 | 63% | 11.5 | 3.05 | 152 | 89 / 155 / 3 / 29 | 18 / 17 | 134 | 16 | 0 | 0 (0) |
| rx2_distant | 150 | 130 | 87% | 11.8 | 3.02 | 183 | 131 / 145 / 2 / 10 | 10 / 8 | 147 | 3 | 0 | 0 (0) |
| all | 625 | 290 | 46% | 11.0 | 2.96 | 593 | | | | | | 0 (0) |
