# Expert baseline_compare_meter_route under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 49 | 98% | 10.6 | 2.90 | 1 | 49 / 42 / 0 / 0 | 0 / 4 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.5 | 2.04 | 18 | 49 / 0 / 0 / 1 | 0 / 2 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 16 | 21% | 13.1 | 3.91 | 26 | 19 / 44 / 43 / 3 | 11 / 98 | 74 | 1 | 0 | 0 (0) |
| meter_rx2 | 150 | 76 | 51% | 15.3 | 3.97 | 145 | 96 / 280 / 28 / 4 | 10 / 27 | 136 | 14 | 0 | 0 (0) |
| rx2_adjacent | 150 | 56 | 37% | 13.7 | 3.48 | 222 | 72 / 184 / 17 / 29 | 45 / 27 | 139 | 11 | 0 | 0 (0) |
| rx2_distant | 150 | 117 | 78% | 11.8 | 3.10 | 114 | 137 / 139 / 2 / 14 | 4 / 21 | 149 | 1 | 0 | 0 (0) |
| all | 625 | 362 | 58% | 12.8 | 3.40 | 526 | | | | | | 0 (0) |
