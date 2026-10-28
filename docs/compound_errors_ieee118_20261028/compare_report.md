# Compound errors across systems (ieee14_study, ieee14_busbranch, ieee118)

## Random-pair geometry

| system | lines | pairs | share a bus | within one hop | >= 3 hops | median / max hops | max bus degree | overlap median at 0 / 1 / 2 / 3+ hops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ieee14_study | 15 | 105 | 23.8% | 54.3% | 7.6% | 1 / 3 | 5 | 0.460 / 0.149 / 0.023 / 0.010 |
| ieee14_busbranch | 15 | 105 | 23.8% | 54.3% | 7.6% | 1 / 3 | 5 | 0.460 / 0.149 / 0.023 / 0.010 |
| ieee118 | 177 | 15576 | 3.4% | 10.3% | 79.7% | 5 / 14 | 12 | 0.185 / 0.060 / 0.018 / 0.000 |

## Per class

| class | system | n | matched (every atom >= floor) | solved, deployed | solved, pair refit | matched solved, deployed / pair refit | mean actions, deployed | silent after one exact fix (matched) | first true fix final with atoms left (deployed) | cue fired |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two R/X lines, adjacent | ieee14_study | 150 | 120 | 50 of 150 (33%) | 76 of 150 (51%) | 48 / 73 of 120 | 16.7 | 25 of 150 (6 of 120) | 20 of 121 | 6 of 150 |
| two R/X lines, adjacent | ieee14_busbranch | 150 | 150 | 61 of 150 (41%) | 86 of 150 (57%) | 61 / 86 of 150 | 17.6 | 8 of 150 (8 of 150) | 9 of 121 | 4 of 150 |
| two R/X lines, adjacent | ieee118 | 150 | 150 | 95 of 150 (63%) | 110 of 150 (73%) | 95 / 110 of 150 | 15.7 | 5 of 150 (5 of 150) | 5 of 130 | 15 of 150 |
| two R/X lines, distant | ieee14_study | 150 | 71 | 72 of 150 (48%) | 72 of 150 (48%) | 56 / 56 of 71 | 10.5 | 58 of 150 (1 of 71) | 57 of 147 | 0 of 150 |
| two R/X lines, distant | ieee14_busbranch | 150 | 150 | 115 of 150 (77%) | 115 of 150 (77%) | 115 / 115 of 150 | 12.4 | 4 of 150 (4 of 150) | 4 of 144 | 1 of 150 |
| two R/X lines, distant | ieee118 | 150 | 150 | 130 of 150 (87%) | 130 of 150 (87%) | 130 / 130 of 150 | 13.0 | 0 of 150 (0 of 150) | 0 of 144 | 16 of 150 |
| meter + two R/X | ieee14_study | 150 | 90 | 61 of 150 (41%) | 66 of 150 (44%) | 54 / 59 of 90 | 16.7 | 1 of 150 (0 of 90) | 1 of 133 | 1 of 150 |
| meter + two R/X | ieee14_busbranch | 150 | 150 | 88 of 150 (59%) | 88 of 150 (59%) | 88 / 88 of 150 | 18.6 | 0 of 150 (0 of 150) | 0 of 132 | 0 of 150 |
| meter + two R/X | ieee118 | 150 | 150 | 130 of 150 (87%) | 130 of 150 (87%) | 130 / 130 of 150 | 17.1 | 0 of 150 (0 of 150) | 0 of 144 | 15 of 150 |
| meter on the faulted line + R/X | ieee14_study | 75 | 52 | 14 of 75 (19%) | 14 of 75 (19%) | 10 / 10 of 52 | 14.7 | 11 of 75 (0 of 52) | 13 of 49 | 0 of 75 |
| meter on the faulted line + R/X | ieee14_busbranch | 75 | 75 | 13 of 75 (17%) | 13 of 75 (17%) | 13 / 13 of 75 | 19.1 | 1 of 75 (1 of 75) | 1 of 31 | 0 of 75 |
| meter on the faulted line + R/X | ieee118 | 75 | 75 | 11 of 75 (15%) | 11 of 75 (15%) | 11 / 11 of 75 | 22.6 | 0 of 75 (0 of 75) | 0 of 26 | 3 of 75 |
| control: one R/X line | ieee14_study | 50 | 45 | 43 of 50 (86%) | 43 of 50 (86%) | 41 / 41 of 45 | 7.5 | n/a | n/a | 0 of 50 |
| control: one R/X line | ieee14_busbranch | 50 | 50 | 48 of 50 (96%) | 48 of 50 (96%) | 48 / 48 of 50 | 7.7 | n/a | n/a | 0 of 50 |
| control: one R/X line | ieee118 | 50 | 50 | 45 of 50 (90%) | 45 of 50 (90%) | 45 / 45 of 50 | 8.0 | n/a | n/a | 4 of 50 |
| control: 1 to 3 meters | ieee14_study | 50 | 50 | 47 of 50 (94%) | 47 of 50 (94%) | 47 / 47 of 50 | 21.3 | 0 of 35 (0 of 35) | 0 of 35 | 0 of 50 |
| control: 1 to 3 meters | ieee14_busbranch | 50 | 50 | 49 of 50 (98%) | 49 of 50 (98%) | 49 / 49 of 50 | 21.6 | 0 of 37 (0 of 37) | 0 of 37 | 0 of 50 |
| control: 1 to 3 meters | ieee118 | 50 | 50 | 45 of 50 (90%) | 45 of 50 (90%) | 45 / 45 of 50 | 21.6 | 0 of 33 (0 of 33) | 0 of 33 | 1 of 50 |

## Ranking ambiguity against the true lines' bus degree

| system | degree | roots | mean lines within the band | band holds two or more | top line is true |
| --- | --- | --- | --- | --- | --- |
| ieee14_study | 1-3 | 43 | 1.88 | 18 | 39 |
| ieee14_study | 4-5 | 532 | 1.75 | 296 | 460 |
| ieee14_busbranch | 1-3 | 10 | 2.10 | 4 | 8 |
| ieee14_busbranch | 4-5 | 565 | 1.86 | 353 | 489 |
| ieee118 | 1-3 | 20 | 2.45 | 13 | 14 |
| ieee118 | 4-5 | 167 | 2.40 | 118 | 150 |
| ieee118 | 6-8 | 293 | 2.13 | 171 | 272 |
| ieee118 | 9+ | 95 | 1.53 | 35 | 93 |
