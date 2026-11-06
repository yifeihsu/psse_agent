# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 28 | 56% | 15.8 | 4.32 | 24 | 28 / 47 / 0 / 0 | 0 / 91 | 49 | 1 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.7 | 2.06 | 24 | 48 / 0 / 0 / 1 | 4 / 0 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 11 | 15% | 14.8 | 4.32 | 30 | 12 / 37 / 55 / 1 | 26 / 118 | 73 | 2 | 0 | 0 (0) |
| meter_rx2 | 150 | 85 | 57% | 16.5 | 4.26 | 157 | 104 / 283 / 25 / 17 | 14 / 46 | 144 | 6 | 0 | 0 (0) |
| rx2_adjacent | 150 | 59 | 39% | 15.2 | 3.82 | 241 | 92 / 195 / 8 / 44 | 46 / 41 | 147 | 3 | 0 | 0 (0) |
| rx2_distant | 150 | 115 | 77% | 12.2 | 3.21 | 104 | 132 / 140 / 7 / 15 | 14 / 23 | 149 | 1 | 0 | 0 (0) |
| all | 625 | 346 | 55% | 14.2 | 3.74 | 580 | | | | | | 0 (0) |
