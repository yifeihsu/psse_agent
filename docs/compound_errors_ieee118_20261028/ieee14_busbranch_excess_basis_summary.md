# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 49 | 98% | 21.6 | 5.70 | 34 | 49 / 55 / 0 / 0 | 0 / 131 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 48 | 96% | 7.7 | 2.06 | 24 | 48 / 0 / 0 / 1 | 4 / 0 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 12 | 16% | 19.4 | 5.52 | 37 | 13 / 43 / 60 / 2 | 53 / 168 | 75 | 0 | 0 | 0 (0) |
| meter_rx2 | 150 | 88 | 59% | 18.7 | 4.83 | 167 | 121 / 300 / 29 / 22 | 26 / 77 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 61 | 41% | 18.0 | 4.52 | 276 | 105 / 205 / 16 / 47 | 85 / 71 | 149 | 1 | 0 | 0 (0) |
| rx2_distant | 150 | 117 | 78% | 12.3 | 3.22 | 104 | 139 / 143 / 2 / 13 | 17 / 19 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 375 | 60% | 16.4 | 4.30 | 642 | | | | | | 0 (0) |
