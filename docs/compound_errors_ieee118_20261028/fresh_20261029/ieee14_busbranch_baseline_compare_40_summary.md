# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 49 | 98% | 17.1 | 4.52 | 19 | 49 / 42 / 0 / 0 | 0 / 85 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.5 | 2.04 | 18 | 49 / 0 / 0 / 1 | 0 / 2 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 17 | 23% | 18.4 | 5.21 | 32 | 19 / 55 / 56 / 6 | 35 / 145 | 74 | 1 | 0 | 0 (0) |
| meter_rx2 | 150 | 88 | 59% | 18.5 | 4.73 | 173 | 118 / 298 / 41 / 10 | 18 / 74 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 60 | 40% | 18.5 | 4.68 | 257 | 92 / 203 / 38 / 35 | 72 / 111 | 149 | 1 | 0 | 0 (0) |
| rx2_distant | 150 | 117 | 78% | 12.6 | 3.31 | 115 | 137 / 140 / 3 / 15 | 16 / 35 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 379 | 61% | 16.1 | 4.20 | 614 | | | | | | 0 (0) |
