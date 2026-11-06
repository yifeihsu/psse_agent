# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 19 | 38% | 12.1 | 3.52 | 21 | 19 / 33 / 0 / 0 | 0 / 74 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 45 | 90% | 7.9 | 2.14 | 28 | 46 / 0 / 0 / 3 | 2 / 6 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 4 | 5% | 14.2 | 4.41 | 43 | 6 / 24 / 59 / 3 | 16 / 151 | 73 | 2 | 0 | 0 (0) |
| meter_rx2 | 150 | 129 | 86% | 15.8 | 4.04 | 199 | 134 / 282 / 2 / 8 | 12 / 19 | 147 | 3 | 0 | 0 (0) |
| rx2_adjacent | 150 | 97 | 65% | 13.0 | 3.41 | 172 | 103 / 164 / 3 / 32 | 22 / 41 | 144 | 6 | 0 | 0 (0) |
| rx2_distant | 150 | 130 | 87% | 12.5 | 3.14 | 183 | 131 / 146 / 2 / 13 | 11 / 21 | 149 | 1 | 0 | 0 (0) |
| all | 625 | 424 | 68% | 13.2 | 3.52 | 646 | | | | | | 0 (0) |
