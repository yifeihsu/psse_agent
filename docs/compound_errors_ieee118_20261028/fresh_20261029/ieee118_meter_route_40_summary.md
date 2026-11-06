# Expert baseline_compare_meter_route under scada_only on 625 compound roots

| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_meter | 50 | 50 | 100% | 11.1 | 3.02 | 0 | 50 / 51 / 0 / 0 | 0 / 0 | 50 | 0 | 0 | 0 (0) |
| control_rx | 50 | 44 | 88% | 8.1 | 2.18 | 27 | 45 / 0 / 0 / 3 | 5 / 6 | 50 | 0 | 0 | 0 (0) |
| meter_on_rx | 75 | 8 | 11% | 22.2 | 6.45 | 60 | 14 / 34 / 67 / 3 | 66 / 225 | 75 | 0 | 0 | 0 (0) |
| meter_rx2 | 150 | 117 | 78% | 18.0 | 4.57 | 199 | 126 / 288 / 17 / 19 | 19 / 68 | 150 | 0 | 0 | 0 (0) |
| rx2_adjacent | 150 | 101 | 67% | 14.3 | 3.73 | 169 | 117 / 164 / 6 / 28 | 33 / 62 | 149 | 1 | 0 | 0 (0) |
| rx2_distant | 150 | 125 | 83% | 13.0 | 3.25 | 188 | 130 / 148 / 1 / 9 | 20 / 29 | 150 | 0 | 0 | 0 (0) |
| all | 625 | 445 | 71% | 15.1 | 3.96 | 643 | | | | | | 0 (0) |
