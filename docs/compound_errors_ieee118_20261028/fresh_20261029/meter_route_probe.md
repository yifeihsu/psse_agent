# Meter-route rule (max normalized residual >= 11) against baseline_compare

## ieee118, 16 actions

| class | n | baseline_compare | rule | gained | lost | mean actions, both solved (baseline / rule) | wrong commits baseline | wrong commits rule | budget handoffs (baseline / rule) | rule fired (episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 150 | 99 | 99 | 0 | 0 | 12.5 / 12.5 | correct_measurements 2, correct_parameters 15 | correct_measurements 2, correct_parameters 15 | 43 / 43 | 0 |
| rx2_distant | 150 | 125 | 125 | 0 | 0 | 12.2 / 12.2 | correct_measurements 2, correct_parameters 10 | correct_measurements 2, correct_parameters 10 | 11 / 11 | 0 |
| meter_rx2 | 150 | 114 | 114 | 0 | 0 | 15.9 / 15.9 | correct_parameters 10, correct_topology 1 | correct_parameters 10, correct_topology 1 | 145 / 145 | 2 |
| meter_on_rx | 75 | 5 | 5 | 0 | 0 | 11.8 / 11.8 | correct_measurements 6, correct_parameters 9, correct_topology 5 | correct_measurements 7, correct_parameters 9, correct_topology 5 | 55 / 55 | 9 |
| control_rx | 50 | 44 | 44 | 0 | 0 | 7.5 / 7.5 | correct_measurements 1, correct_parameters 3 | correct_measurements 1, correct_parameters 3 | 2 / 2 | 0 |
| control_meter | 50 | 18 | 50 | 32 | 0 | 8.0 / 7.4 | 0 | 0 | 34 / 17 | 34 |
| all | 625 | 405 | 437 | 32 | 0 | 12.6 / 12.6 | correct_measurements 11, correct_parameters 47, correct_topology 6 | correct_measurements 12, correct_parameters 47, correct_topology 6 | 290 / 273 | 45 |

Trajectories the rule changes: meter_rx2 2 (+0 actions), meter_on_rx 8 (-1 actions), control_meter 34 (-63 actions).  Roots where baseline_compare never reaches a firing state: 580, of them changed: 0.

baseline_compare's first committed correction with the alarm on, by the next true correction (section 9.4):

| committed -> next true | n | median max normalized residual | at least 11 |
| --- | --- | --- | --- |
| line -> line | 324 | 14.5 | 221 |
| line -> meter | 36 | 22.8 | 36 |
| line -> none | 53 | 14.6 | 39 |
| meter -> line | 21 | 7.3 | 0 |
| meter -> meter | 2 | 15.8 | 2 |
| meter -> none | 47 | 16.7 | 38 |

Next true correction after each committed meter correction with the alarm on:

| run | rule fires: meter / line / none | rule quiet: meter / line / none |
| --- | --- | --- |
| baseline | 2 / 1 / 43 | 0 / 48 / 17 |
| baseline, after a true meter commit | 2 / 1 / 37 | 0 / 48 / 16 |
| baseline, after a healthy meter commit | 0 / 0 / 6 | 0 / 0 / 1 |
| rule | 51 / 1 / 11 | 0 / 48 / 18 |
| rule, after a true meter commit | 51 / 1 / 5 | 0 / 48 / 16 |
| rule, after a healthy meter commit | 0 / 0 / 6 | 0 / 0 / 2 |

## ieee118, 40 actions

| class | n | baseline_compare | rule | gained | lost | mean actions, both solved (baseline / rule) | wrong commits baseline | wrong commits rule | budget handoffs (baseline / rule) | rule fired (episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 150 | 101 | 101 | 0 | 0 | 13.1 / 13.1 | correct_measurements 8, correct_parameters 23, correct_topology 2 | correct_measurements 8, correct_parameters 23, correct_topology 2 | 3 / 3 | 0 |
| rx2_distant | 150 | 125 | 125 | 0 | 0 | 12.2 / 12.2 | correct_measurements 5, correct_parameters 13, correct_topology 1 | correct_measurements 6, correct_parameters 13, correct_topology 1 | 2 / 2 | 2 |
| meter_rx2 | 150 | 117 | 117 | 0 | 0 | 16.7 / 16.7 | correct_measurements 3, correct_parameters 12, correct_topology 4 | correct_measurements 3, correct_parameters 12, correct_topology 4 | 4 / 4 | 6 |
| meter_on_rx | 75 | 8 | 8 | 0 | 0 | 16.1 / 16.1 | correct_measurements 34, correct_parameters 12, correct_topology 16 | correct_measurements 39, correct_parameters 11, correct_topology 16 | 13 / 8 | 24 |
| control_rx | 50 | 44 | 44 | 0 | 0 | 7.5 / 7.5 | correct_measurements 2, correct_parameters 3 | correct_measurements 2, correct_parameters 3 | 0 / 0 | 0 |
| control_meter | 50 | 46 | 50 | 4 | 0 | 20.4 / 10.7 | 0 | 0 | 9 / 0 | 34 |
| all | 625 | 441 | 445 | 4 | 0 | 14.1 / 13.1 | correct_measurements 52, correct_parameters 63, correct_topology 23 | correct_measurements 58, correct_parameters 62, correct_topology 23 | 31 / 17 | 66 |

Trajectories the rule changes: rx2_distant 1 (+0 actions), meter_rx2 5 (-2 actions), meter_on_rx 21 (-31 actions), control_meter 34 (-543 actions).  Roots where baseline_compare never reaches a firing state: 559, of them changed: 0.

baseline_compare's first committed correction with the alarm on, by the next true correction (section 9.4):

| committed -> next true | n | median max normalized residual | at least 11 |
| --- | --- | --- | --- |
| line -> line | 327 | 14.5 | 224 |
| line -> meter | 45 | 22.8 | 45 |
| line -> none | 41 | 13.7 | 27 |
| meter -> line | 28 | 7.5 | 3 |
| meter -> meter | 34 | 19.0 | 34 |
| meter -> none | 15 | 11.2 | 8 |

Next true correction after each committed meter correction with the alarm on:

| run | rule fires: meter / line / none | rule quiet: meter / line / none |
| --- | --- | --- |
| baseline | 48 / 7 / 38 | 0 / 55 / 33 |
| baseline, after a true meter commit | 47 / 7 / 10 | 0 / 55 / 18 |
| baseline, after a healthy meter commit | 1 / 0 / 28 | 0 / 0 / 15 |
| rule | 52 / 6 / 37 | 0 / 55 / 35 |
| rule, after a true meter commit | 51 / 6 / 7 | 0 / 55 / 18 |
| rule, after a healthy meter commit | 1 / 0 / 30 | 0 / 0 / 17 |

## ieee14_control, 16 actions

| class | n | baseline_compare | rule | gained | lost | mean actions, both solved (baseline / rule) | wrong commits baseline | wrong commits rule | budget handoffs (baseline / rule) | rule fired (episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 150 | 56 | 56 | 0 | 0 | 13.9 / 13.9 | correct_measurements 13, correct_parameters 26, correct_topology 5 | correct_measurements 14, correct_parameters 26, correct_topology 5 | 80 / 80 | 5 |
| rx2_distant | 150 | 117 | 117 | 0 | 0 | 11.8 / 11.8 | correct_parameters 3, correct_topology 1 | correct_parameters 3, correct_topology 1 | 9 / 9 | 0 |
| meter_rx2 | 150 | 77 | 76 | 0 | 1 | 15.7 / 15.7 | correct_measurements 1, correct_parameters 5, correct_topology 2 | correct_measurements 3, correct_parameters 5, correct_topology 2 | 127 / 127 | 7 |
| meter_on_rx | 75 | 16 | 16 | 0 | 0 | 11.3 / 11.3 | correct_measurements 4, correct_parameters 4, correct_topology 1 | correct_measurements 6, correct_parameters 4, correct_topology 1 | 39 / 39 | 5 |
| control_rx | 50 | 48 | 48 | 0 | 0 | 7.4 / 7.4 | 0 | 0 | 0 / 0 | 0 |
| control_meter | 50 | 27 | 49 | 22 | 0 | 9.6 / 8.7 | 0 | 0 | 30 / 13 | 27 |
| all | 625 | 341 | 362 | 22 | 1 | 12.2 / 12.1 | correct_measurements 18, correct_parameters 38, correct_topology 9 | correct_measurements 23, correct_parameters 38, correct_topology 9 | 285 / 268 | 44 |

Trajectories the rule changes: rx2_adjacent 6 (-1 actions), meter_rx2 7 (+2 actions), meter_on_rx 4 (-1 actions), control_meter 27 (-62 actions).  Roots where baseline_compare never reaches a firing state: 581, of them changed: 1 (r0_a590aacb3bc1).

baseline_compare's first committed correction with the alarm on, by the next true correction (section 9.4):

| committed -> next true | n | median max normalized residual | at least 11 |
| --- | --- | --- | --- |
| line -> line | 296 | 9.4 | 126 |
| line -> meter | 51 | 19.4 | 48 |
| line -> none | 40 | 12.7 | 20 |
| meter -> line | 46 | 6.8 | 4 |
| meter -> meter | 10 | 12.7 | 6 |
| meter -> none | 38 | 13.3 | 24 |

Next true correction after each committed meter correction with the alarm on:

| run | rule fires: meter / line / none | rule quiet: meter / line / none |
| --- | --- | --- |
| baseline | 6 / 8 / 30 | 4 / 81 / 41 |
| baseline, after a true meter commit | 6 / 8 / 22 | 4 / 80 / 35 |
| baseline, after a healthy meter commit | 0 / 0 / 8 | 0 / 1 / 6 |
| rule | 36 / 6 / 12 | 6 / 81 / 44 |
| rule, after a true meter commit | 35 / 6 / 3 | 6 / 80 / 35 |
| rule, after a healthy meter commit | 1 / 0 / 9 | 0 / 1 / 9 |

## ieee14_control, 40 actions

| class | n | baseline_compare | rule | gained | lost | mean actions, both solved (baseline / rule) | wrong commits baseline | wrong commits rule | budget handoffs (baseline / rule) | rule fired (episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 150 | 60 | 60 | 0 | 0 | 15.8 / 15.8 | correct_measurements 25, correct_parameters 40, correct_topology 7 | correct_measurements 26, correct_parameters 40, correct_topology 7 | 6 / 4 | 10 |
| rx2_distant | 150 | 117 | 117 | 0 | 0 | 11.8 / 11.8 | correct_measurements 5, correct_parameters 3, correct_topology 8 | correct_measurements 5, correct_parameters 3, correct_topology 8 | 0 / 0 | 0 |
| meter_rx2 | 150 | 88 | 87 | 0 | 1 | 17.2 / 17.2 | correct_measurements 5, correct_parameters 9, correct_topology 4 | correct_measurements 7, correct_parameters 9, correct_topology 4 | 3 / 3 | 8 |
| meter_on_rx | 75 | 17 | 17 | 0 | 0 | 11.9 / 11.9 | correct_measurements 24, correct_parameters 5, correct_topology 6 | correct_measurements 27, correct_parameters 4, correct_topology 6 | 6 / 6 | 16 |
| control_rx | 50 | 48 | 48 | 0 | 0 | 7.4 / 7.4 | 0 | 0 | 0 / 0 | 0 |
| control_meter | 50 | 49 | 50 | 1 | 0 | 16.7 / 10.8 | 0 | 0 | 2 / 0 | 27 |
| all | 625 | 379 | 379 | 1 | 1 | 13.8 / 13.0 | correct_measurements 59, correct_parameters 57, correct_topology 25 | correct_measurements 65, correct_parameters 56, correct_topology 25 | 17 / 13 | 61 |

Trajectories the rule changes: rx2_adjacent 8 (-31 actions), meter_rx2 8 (+2 actions), meter_on_rx 13 (-25 actions), control_meter 27 (-313 actions).  Roots where baseline_compare never reaches a firing state: 564, of them changed: 0.

baseline_compare's first committed correction with the alarm on, by the next true correction (section 9.4):

| committed -> next true | n | median max normalized residual | at least 11 |
| --- | --- | --- | --- |
| line -> line | 303 | 9.4 | 130 |
| line -> meter | 54 | 19.4 | 51 |
| line -> none | 30 | 8.5 | 13 |
| meter -> line | 52 | 7.0 | 5 |
| meter -> meter | 33 | 13.8 | 28 |
| meter -> none | 17 | 8.2 | 5 |

Next true correction after each committed meter correction with the alarm on:

| run | rule fires: meter / line / none | rule quiet: meter / line / none |
| --- | --- | --- |
| baseline | 36 / 12 / 27 | 8 / 100 / 46 |
| baseline, after a true meter commit | 34 / 10 / 5 | 7 / 97 / 25 |
| baseline, after a healthy meter commit | 2 / 2 / 22 | 1 / 3 / 21 |
| rule | 37 / 9 / 31 | 8 / 100 / 50 |
| rule, after a true meter commit | 35 / 7 / 7 | 7 / 97 / 25 |
| rule, after a healthy meter commit | 2 / 2 / 24 | 1 / 3 / 25 |

## Against the reference probe

### ieee118

| class | 16 actions, reference | 16 actions, this draw | 40 actions, reference | 40 actions, this draw |
| --- | --- | --- | --- | --- |
| rx2_adjacent (150) | 97 / 97 (+0, -0) | 99 / 99 (+0, -0) | 95 / 95 (+0, -0) | 101 / 101 (+0, -0) |
| rx2_distant (150) | 130 / 130 (+0, -0) | 125 / 125 (+0, -0) | 130 / 130 (+0, -0) | 125 / 125 (+0, -0) |
| meter_rx2 (150) | 129 / 129 (+0, -0) | 114 / 114 (+0, -0) | 130 / 130 (+0, -0) | 117 / 117 (+0, -0) |
| meter_on_rx (75) | 4 / 4 (+0, -0) | 5 / 5 (+0, -0) | 11 / 11 (+0, -0) | 8 / 8 (+0, -0) |
| control_rx (50) | 45 / 45 (+0, -0) | 44 / 44 (+0, -0) | 45 / 45 (+0, -0) | 44 / 44 (+0, -0) |
| control_meter (50) | 19 / 49 (+30, -0) | 18 / 50 (+32, -0) | 45 / 50 (+5, -0) | 46 / 50 (+4, -0) |
| all (625) | 424 / 454 (+30, -0) | 405 / 437 (+32, -0) | 456 / 461 (+5, -0) | 441 / 445 (+4, -0) |

### ieee14_control

| class | 16 actions, reference | 16 actions, this draw | 40 actions, reference | 40 actions, this draw |
| --- | --- | --- | --- | --- |
| rx2_adjacent (150) | 59 / 59 (+0, -0) | 56 / 56 (+0, -0) | 61 / 61 (+0, -0) | 60 / 60 (+0, -0) |
| rx2_distant (150) | 115 / 115 (+0, -0) | 117 / 117 (+0, -0) | 115 / 115 (+0, -0) | 117 / 117 (+0, -0) |
| meter_rx2 (150) | 77 / 77 (+0, -0) | 77 / 76 (+0, -1) | 88 / 88 (+0, -0) | 88 / 87 (+0, -1) |
| meter_on_rx (75) | 11 / 11 (+0, -0) | 16 / 16 (+0, -0) | 13 / 13 (+0, -0) | 17 / 17 (+0, -0) |
| control_rx (50) | 48 / 48 (+0, -0) | 48 / 48 (+0, -0) | 48 / 48 (+0, -0) | 48 / 48 (+0, -0) |
| control_meter (50) | 18 / 46 (+28, -0) | 27 / 49 (+22, -0) | 49 / 50 (+1, -0) | 49 / 50 (+1, -0) |
| all (625) | 328 / 356 (+28, -0) | 341 / 362 (+22, -1) | 374 / 375 (+1, -0) | 379 / 379 (+1, -1) |

