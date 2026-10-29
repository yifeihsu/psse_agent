# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 25 | 50% | 15.2 | 4.20 | 21 | 25 / 44 / 0 / 0 | 0 / 91 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 45 | 90% | 8.0 | 2.16 | 28 | 46 / 0 / 0 / 3 | 3 / 6 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 9 | 12% | 16.9 | 5.01 | 50 | 12 / 30 / 60 / 3 | 29 / 167 | 73 | 2 | 0 | 0 (0) |
| meter_rx2 | 150 | 129 | 86% | 16.5 | 4.12 | 201 | 135 / 285 / 2 / 9 | 12 / 25 | 148 | 2 | 0 | 0 (0) |
| rx2_adjacent | 150 | 96 | 64% | 14.0 | 3.67 | 180 | 110 / 170 / 3 / 33 | 28 / 58 | 149 | 1 | 0 | 0 (0) |
| rx2_distant | 150 | 130 | 87% | 12.8 | 3.22 | 183 | 131 / 146 / 2 / 13 | 15 / 26 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 434 | 69% | 14.3 | 3.75 | 663 | | | | | | 0 (0) |
