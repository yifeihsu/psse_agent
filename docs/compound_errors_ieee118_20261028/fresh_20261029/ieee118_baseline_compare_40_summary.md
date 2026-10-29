# Expert baseline_compare under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 46 | 92% | 21.9 | 5.92 | 26 | 46 / 51 / 0 / 0 | 0 / 149 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 44 | 88% | 8.1 | 2.18 | 27 | 45 / 0 / 0 / 3 | 5 / 6 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 8 | 11% | 22.6 | 6.48 | 63 | 14 / 35 / 70 / 4 | 62 / 226 | 75 | 0 | 0 | 0 (0) |
| meter_rx2 | 150 | 117 | 78% | 18.0 | 4.57 | 199 | 126 / 288 / 17 / 18 | 19 / 68 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 101 | 67% | 14.3 | 3.73 | 169 | 117 / 164 / 6 / 28 | 33 / 62 | 149 | 1 | 0 | 0 (0) |
| rx2_distant | 150 | 125 | 83% | 13.0 | 3.25 | 189 | 130 / 148 / 1 / 10 | 19 / 30 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 441 | 71% | 16.0 | 4.20 | 673 | | | | | | 0 (0) |
