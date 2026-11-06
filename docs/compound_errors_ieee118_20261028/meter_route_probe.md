# Meter-route rule (max normalized residual >= 11) against baseline_compare

## ieee118, 16 actions

| class | n | baseline_compare | rule | gained | lost | mean actions, both solved (baseline / rule) | wrong commits baseline | wrong commits rule | budget handoffs (baseline / rule) | rule fired (episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 150 | 97 | 97 | 0 | 0 | 12.8 / 12.8 | correct_parameters 15, correct_topology 7 | correct_parameters 15, correct_topology 7 | 49 / 49 | 0 |
| rx2_distant | 150 | 130 | 130 | 0 | 0 | 12.2 / 12.2 | correct_measurements 1, correct_parameters 10 | correct_measurements 1, correct_parameters 10 | 13 / 13 | 0 |
| meter_rx2 | 150 | 129 | 129 | 0 | 0 | 15.9 / 15.9 | correct_parameters 10, correct_topology 2 | correct_parameters 10, correct_topology 2 | 146 / 146 | 3 |
| meter_on_rx | 75 | 4 | 4 | 0 | 0 | 11.8 / 11.8 | correct_measurements 4, correct_parameters 6, correct_topology 6 | correct_measurements 5, correct_parameters 6, correct_topology 6 | 60 / 60 | 9 |
| control_rx | 50 | 45 | 45 | 0 | 0 | 7.6 / 7.6 | correct_parameters 2 | correct_parameters 2 | 2 / 2 | 0 |
| control_meter | 50 | 19 | 49 | 30 | 0 | 7.9 / 7.4 | 0 | 0 | 33 / 18 | 33 |
| all | 625 | 424 | 454 | 30 | 0 | 12.8 / 12.8 | correct_measurements 5, correct_parameters 43, correct_topology 15 | correct_measurements 6, correct_parameters 43, correct_topology 15 | 303 / 288 | 45 |

Trajectories the rule changes: meter_rx2 3 (+0 actions), meter_on_rx 8 (+1 actions), control_meter 33 (-53 actions).  Roots where baseline_compare never reaches a firing state: 580, of them changed: 0.

baseline_compare's first committed correction with the alarm on, by the next true correction (section 9.4):

| committed -> next true | n | median max normalized residual | at least 11 |
| --- | --- | --- | --- |
| line -> line | 344 | 14.6 | 233 |
| line -> meter | 39 | 19.8 | 39 |
| line -> none | 41 | 12.2 | 23 |
| meter -> line | 10 | 9.7 | 1 |
| meter -> meter | 1 | 19.4 | 1 |
| meter -> none | 42 | 16.8 | 36 |

Next true correction after each committed meter correction with the alarm on:

| run | rule fires: meter / line / none | rule quiet: meter / line / none |
| --- | --- | --- |
| baseline | 2 / 4 / 40 | 0 / 36 / 12 |
| baseline, after a true meter commit | 2 / 4 / 36 | 0 / 36 / 12 |
| baseline, after a healthy meter commit | 0 / 0 / 4 | 0 / 0 / 0 |
| rule | 50 / 4 / 9 | 0 / 36 / 14 |
| rule, after a true meter commit | 50 / 4 / 5 | 0 / 36 / 13 |
| rule, after a healthy meter commit | 0 / 0 / 4 | 0 / 0 / 1 |

## ieee118, 40 actions

| class | n | baseline_compare | rule | gained | lost | mean actions, both solved (baseline / rule) | wrong commits baseline | wrong commits rule | budget handoffs (baseline / rule) | rule fired (episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 150 | 95 | 95 | 0 | 0 | 13.2 / 13.2 | correct_measurements 10, correct_parameters 23, correct_topology 12 | correct_measurements 11, correct_parameters 23, correct_topology 12 | 4 / 4 | 4 |
| rx2_distant | 150 | 130 | 130 | 0 | 0 | 12.2 / 12.2 | correct_measurements 9, correct_parameters 11, correct_topology 1 | correct_measurements 9, correct_parameters 11, correct_topology 1 | 1 / 1 | 0 |
| meter_rx2 | 150 | 130 | 130 | 0 | 0 | 16.5 / 16.5 | correct_measurements 3, correct_parameters 10, correct_topology 3 | correct_measurements 3, correct_parameters 10, correct_topology 3 | 2 / 2 | 4 |
| meter_on_rx | 75 | 11 | 11 | 0 | 0 | 18.1 / 18.1 | correct_measurements 25, correct_parameters 9, correct_topology 16 | correct_measurements 30, correct_parameters 9, correct_topology 16 | 8 / 5 | 21 |
| control_rx | 50 | 45 | 45 | 0 | 0 | 7.6 / 7.6 | correct_measurements 1, correct_parameters 2 | correct_measurements 1, correct_parameters 2 | 0 / 0 | 0 |
| control_meter | 50 | 45 | 50 | 5 | 0 | 19.8 / 10.8 | 0 | 0 | 7 / 0 | 33 |
| all | 625 | 456 | 461 | 5 | 0 | 14.1 / 13.2 | correct_measurements 48, correct_parameters 55, correct_topology 32 | correct_measurements 54, correct_parameters 55, correct_topology 32 | 22 / 12 | 62 |

Trajectories the rule changes: rx2_adjacent 4 (-7 actions), meter_rx2 4 (-2 actions), meter_on_rx 18 (-51 actions), control_meter 33 (-522 actions).  Roots where baseline_compare never reaches a firing state: 563, of them changed: 0.

baseline_compare's first committed correction with the alarm on, by the next true correction (section 9.4):

| committed -> next true | n | median max normalized residual | at least 11 |
| --- | --- | --- | --- |
| line -> line | 347 | 14.5 | 234 |
| line -> meter | 44 | 19.8 | 44 |
| line -> none | 33 | 11.5 | 17 |
| meter -> line | 19 | 9.6 | 4 |
| meter -> meter | 32 | 19.4 | 32 |
| meter -> none | 8 | 11.4 | 4 |

Next true correction after each committed meter correction with the alarm on:

| run | rule fires: meter / line / none | rule quiet: meter / line / none |
| --- | --- | --- |
| baseline | 46 / 7 / 31 | 1 / 43 / 25 |
| baseline, after a true meter commit | 45 / 7 / 9 | 1 / 43 / 16 |
| baseline, after a healthy meter commit | 1 / 0 / 22 | 0 / 0 / 9 |
| rule | 56 / 7 / 26 | 1 / 43 / 27 |
| rule, after a true meter commit | 50 / 7 / 4 | 1 / 43 / 17 |
| rule, after a healthy meter commit | 6 / 0 / 22 | 0 / 0 / 10 |

## ieee14_control, 16 actions

| class | n | baseline_compare | rule | gained | lost | mean actions, both solved (baseline / rule) | wrong commits baseline | wrong commits rule | budget handoffs (baseline / rule) | rule fired (episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 150 | 59 | 59 | 0 | 0 | 14.2 / 14.2 | correct_measurements 4, correct_parameters 23, correct_topology 9 | correct_measurements 4, correct_parameters 23, correct_topology 9 | 80 / 80 | 1 |
| rx2_distant | 150 | 115 | 115 | 0 | 0 | 11.7 / 11.7 | correct_measurements 1, correct_parameters 6, correct_topology 1 | correct_measurements 1, correct_parameters 6, correct_topology 1 | 14 / 14 | 1 |
| meter_rx2 | 150 | 77 | 77 | 0 | 0 | 15.8 / 15.8 | correct_measurements 2, correct_parameters 6, correct_topology 3 | correct_measurements 3, correct_parameters 6, correct_topology 3 | 126 / 126 | 6 |
| meter_on_rx | 75 | 11 | 11 | 0 | 0 | 11.3 / 11.3 | correct_measurements 2, correct_parameters 8, correct_topology 6 | correct_measurements 2, correct_parameters 8, correct_topology 6 | 41 / 41 | 1 |
| control_rx | 50 | 48 | 48 | 0 | 0 | 7.4 / 7.4 | correct_parameters 3 | correct_parameters 3 | 1 / 1 | 0 |
| control_meter | 50 | 18 | 46 | 28 | 0 | 9.1 / 8.6 | 0 | 0 | 36 / 18 | 36 |
| all | 625 | 328 | 356 | 28 | 0 | 12.3 / 12.3 | correct_measurements 9, correct_parameters 46, correct_topology 19 | correct_measurements 10, correct_parameters 46, correct_topology 19 | 298 / 280 | 45 |

Trajectories the rule changes: rx2_adjacent 1 (+0 actions), rx2_distant 1 (+0 actions), meter_rx2 6 (+1 actions), meter_on_rx 1 (+0 actions), control_meter 33 (-57 actions).  Roots where baseline_compare never reaches a firing state: 580, of them changed: 0.

baseline_compare's first committed correction with the alarm on, by the next true correction (section 9.4):

| committed -> next true | n | median max normalized residual | at least 11 |
| --- | --- | --- | --- |
| line -> line | 282 | 9.5 | 126 |
| line -> meter | 59 | 17.6 | 49 |
| line -> none | 47 | 9.6 | 18 |
| meter -> line | 34 | 7.4 | 3 |
| meter -> meter | 5 | 12.8 | 5 |
| meter -> none | 44 | 15.2 | 33 |

Next true correction after each committed meter correction with the alarm on:

| run | rule fires: meter / line / none | rule quiet: meter / line / none |
| --- | --- | --- |
| baseline | 5 / 3 / 37 | 2 / 81 / 33 |
| baseline, after a true meter commit | 5 / 3 / 33 | 2 / 80 / 30 |
| baseline, after a healthy meter commit | 0 / 0 / 4 | 0 / 1 / 3 |
| rule | 49 / 3 / 5 | 3 / 81 / 37 |
| rule, after a true meter commit | 47 / 3 / 2 | 3 / 80 / 34 |
| rule, after a healthy meter commit | 2 / 0 / 3 | 0 / 1 / 3 |

## ieee14_control, 40 actions

| class | n | baseline_compare | rule | gained | lost | mean actions, both solved (baseline / rule) | wrong commits baseline | wrong commits rule | budget handoffs (baseline / rule) | rule fired (episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 150 | 61 | 61 | 0 | 0 | 16.3 / 16.3 | correct_measurements 17, correct_parameters 36, correct_topology 9 | correct_measurements 18, correct_parameters 36, correct_topology 9 | 8 / 7 | 4 |
| rx2_distant | 150 | 115 | 115 | 0 | 0 | 11.7 / 11.7 | correct_measurements 2, correct_parameters 6, correct_topology 8 | correct_measurements 2, correct_parameters 6, correct_topology 8 | 0 / 0 | 1 |
| meter_rx2 | 150 | 88 | 88 | 0 | 0 | 17.2 / 17.2 | correct_measurements 5, correct_parameters 9, correct_topology 8 | correct_measurements 8, correct_parameters 9, correct_topology 8 | 9 / 7 | 8 |
| meter_on_rx | 75 | 13 | 13 | 0 | 0 | 13.0 / 13.0 | correct_measurements 25, correct_parameters 8, correct_topology 13 | correct_measurements 29, correct_parameters 8, correct_topology 13 | 8 / 6 | 14 |
| control_rx | 50 | 48 | 48 | 0 | 0 | 7.4 / 7.4 | correct_parameters 4 | correct_parameters 4 | 0 / 0 | 0 |
| control_meter | 50 | 49 | 50 | 1 | 0 | 21.3 / 12.4 | 0 | 0 | 7 / 1 | 36 |
| all | 625 | 374 | 375 | 1 | 0 | 14.5 / 13.3 | correct_measurements 49, correct_parameters 63, correct_topology 38 | correct_measurements 57, correct_parameters 63, correct_topology 38 | 32 / 21 | 63 |

Trajectories the rule changes: rx2_adjacent 3 (-6 actions), rx2_distant 1 (-2 actions), meter_rx2 8 (-7 actions), meter_on_rx 11 (-15 actions), control_meter 33 (-460 actions).  Roots where baseline_compare never reaches a firing state: 562, of them changed: 0.

baseline_compare's first committed correction with the alarm on, by the next true correction (section 9.4):

| committed -> next true | n | median max normalized residual | at least 11 |
| --- | --- | --- | --- |
| line -> line | 287 | 9.5 | 130 |
| line -> meter | 64 | 17.3 | 54 |
| line -> none | 37 | 7.2 | 9 |
| meter -> line | 36 | 7.4 | 4 |
| meter -> meter | 39 | 15.5 | 37 |
| meter -> none | 12 | 7.5 | 2 |

Next true correction after each committed meter correction with the alarm on:

| run | rule fires: meter / line / none | rule quiet: meter / line / none |
| --- | --- | --- |
| baseline | 49 / 4 / 31 | 9 / 105 / 33 |
| baseline, after a true meter commit | 46 / 4 / 6 | 9 / 104 / 17 |
| baseline, after a healthy meter commit | 3 / 0 / 25 | 0 / 1 / 16 |
| rule | 52 / 4 / 32 | 9 / 105 / 38 |
| rule, after a true meter commit | 47 / 4 / 6 | 9 / 104 / 17 |
| rule, after a healthy meter commit | 5 / 0 / 26 | 0 / 1 / 21 |

