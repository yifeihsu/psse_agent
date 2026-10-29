# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 27 | 54% | 11.9 | 3.34 | 16 | 27 / 33 / 0 / 0 | 0 / 57 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.5 | 2.04 | 18 | 49 / 0 / 0 / 1 | 0 / 2 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 16 | 21% | 13.1 | 3.92 | 26 | 19 / 44 / 44 / 3 | 9 / 100 | 74 | 1 | 0 | 0 (0) |
| meter_rx2 | 150 | 77 | 51% | 15.3 | 3.96 | 146 | 97 / 280 / 28 / 5 | 8 / 27 | 136 | 14 | 0 | 0 (0) |
| rx2_adjacent | 150 | 56 | 37% | 13.7 | 3.49 | 221 | 72 / 184 / 18 / 29 | 44 / 28 | 139 | 11 | 0 | 0 (0) |
| rx2_distant | 150 | 117 | 78% | 11.8 | 3.10 | 114 | 137 / 139 / 2 / 14 | 4 / 21 | 149 | 1 | 0 | 0 (0) |
| all | 625 | 341 | 55% | 12.9 | 3.43 | 541 | | | | | | 0 (0) |
