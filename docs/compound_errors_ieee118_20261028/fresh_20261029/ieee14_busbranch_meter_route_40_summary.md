# Expert baseline_compare_meter_route under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 50 | 100% | 10.8 | 2.96 | 1 | 50 / 42 / 0 / 0 | 0 / 6 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.5 | 2.04 | 18 | 49 / 0 / 0 / 1 | 0 / 2 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 17 | 23% | 18.1 | 5.17 | 31 | 19 / 54 / 52 / 6 | 37 / 143 | 73 | 2 | 0 | 0 (0) |
| meter_rx2 | 150 | 87 | 58% | 18.5 | 4.72 | 175 | 117 / 297 / 42 / 8 | 20 / 74 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 60 | 40% | 18.3 | 4.64 | 257 | 92 / 203 / 32 / 34 | 73 / 107 | 145 | 5 | 0 | 0 (0) |
| rx2_distant | 150 | 117 | 78% | 12.6 | 3.31 | 115 | 137 / 140 / 3 / 15 | 16 / 35 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 379 | 61% | 15.5 | 4.06 | 597 | | | | | | 0 (0) |
