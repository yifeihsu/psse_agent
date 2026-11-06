# Success under an action budget

## ieee118

| class | 12 actions: solved (ran to budget / budget handoffs) | 16 actions: solved (ran to budget / budget handoffs) | 20 actions: solved (ran to budget / budget handoffs) | 40 actions: solved (ran to budget / budget handoffs) |
| --- | --- | --- | --- | --- |
| rx2_adjacent | 94 of 150 (16 / 129) | 97 of 150 (6 / 49) | 96 of 150 (1 / 37) | 95 of 150 (0 / 4) |
| rx2_distant | 130 of 150 (3 / 147) | 130 of 150 (1 / 13) | 130 of 150 (0 / 10) | 130 of 150 (0 / 1) |
| meter_rx2 | 0 of 150 (50 / 100) | 129 of 150 (3 / 146) | 129 of 150 (2 / 9) | 130 of 150 (0 / 2) |
| meter_on_rx | 4 of 75 (1 / 74) | 4 of 75 (2 / 60) | 9 of 75 (2 / 45) | 11 of 75 (0 / 8) |
| control_rx | 45 of 50 (0 / 3) | 45 of 50 (0 / 2) | 45 of 50 (0 / 1) | 45 of 50 (0 / 0) |
| control_meter | 17 of 50 (0 / 33) | 19 of 50 (0 / 33) | 25 of 50 (0 / 31) | 45 of 50 (0 / 7) |
| all | 290 of 625 (70 / 486) | 424 of 625 (12 / 303) | 434 of 625 (5 / 133) | 456 of 625 (0 / 22) |

Roots lost at 16 actions (solved at 40), audited under 16:

| class | lost | legal alternative recovers | truth probe only | neither | solving alternatives | mean actions at 40 | mean rollbacks at 40 | mean truth-probe actions (solved) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 0 | 0 | 0 | 0 |  |  |  |  |
| rx2_distant | 0 | 0 | 0 | 0 |  |  |  |  |
| meter_rx2 | 1 | 0 | 0 | 1 |  | 30.0 | 4.0 |  |
| meter_on_rx | 7 | 0 | 7 | 0 |  | 21.7 | 2.0 | 12.0 |
| control_rx | 0 | 0 | 0 | 0 |  |  |  |  |
| control_meter | 26 | 1 | 23 | 2 | {'correct_measurements': 1} | 28.5 | 3.9 | 13.8 |
| all | 34 | 1 | 30 | 3 | {'correct_measurements': 1} | 27.1 | 3.5 | 13.4 |

Max normalized residual after the first committed correction, alarm still on (40 actions), by committed correction -> next true correction:

| committed -> next true | n | min | p10 | median | p90 | share >= 10 |
| --- | --- | --- | --- | --- | --- | --- |
| correct_parameters -> correct_parameters | 347 | 4.6 | 7.7 | 14.5 | 31.2 | 75% |
| correct_parameters -> correct_measurements | 44 | 11.3 | 14.8 | 19.8 | 27.0 | 100% |
| correct_parameters -> none | 33 | 5.6 | 7.0 | 11.5 | 22.5 | 55% |
| correct_measurements -> correct_measurements | 32 | 12.9 | 14.9 | 19.4 | 21.5 | 100% |
| correct_measurements -> correct_parameters | 19 | 5.3 | 6.2 | 9.6 | 11.4 | 42% |
| correct_topology -> none | 15 | 11.1 | 12.3 | 24.6 | 47.4 | 100% |
| correct_measurements -> none | 8 | 7.7 | 7.7 | 11.4 | 99.5 | 62% |
| correct_topology -> correct_parameters | 7 | 14.7 | 14.7 | 23.5 | 35.4 | 100% |
| correct_topology -> correct_measurements | 3 | 13.5 | 13.5 | 15.4 | 22.3 | 100% |

## ieee14_control

| class | 12 actions: solved (ran to budget / budget handoffs) | 16 actions: solved (ran to budget / budget handoffs) | 20 actions: solved (ran to budget / budget handoffs) | 40 actions: solved (ran to budget / budget handoffs) |
| --- | --- | --- | --- | --- |
| rx2_adjacent | 44 of 150 (46 / 95) | 59 of 150 (11 / 80) | 59 of 150 (3 / 56) | 61 of 150 (0 / 8) |
| rx2_distant | 115 of 150 (1 / 145) | 115 of 150 (1 / 14) | 115 of 150 (1 / 9) | 115 of 150 (0 / 0) |
| meter_rx2 | 0 of 150 (11 / 139) | 77 of 150 (12 / 126) | 85 of 150 (6 / 40) | 88 of 150 (0 / 9) |
| meter_on_rx | 11 of 75 (1 / 72) | 11 of 75 (1 / 41) | 11 of 75 (2 / 26) | 13 of 75 (0 / 8) |
| control_rx | 48 of 50 (0 / 1) | 48 of 50 (0 / 1) | 48 of 50 (0 / 1) | 48 of 50 (0 / 0) |
| control_meter | 14 of 50 (0 / 37) | 18 of 50 (0 / 36) | 28 of 50 (1 / 31) | 49 of 50 (0 / 7) |
| all | 232 of 625 (59 / 489) | 328 of 625 (25 / 298) | 346 of 625 (13 / 163) | 374 of 625 (0 / 32) |

Roots lost at 16 actions (solved at 40), audited under 16:

| class | lost | legal alternative recovers | truth probe only | neither | solving alternatives | mean actions at 40 | mean rollbacks at 40 | mean truth-probe actions (solved) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx2_adjacent | 6 | 4 | 0 | 2 | {'correct_parameters': 4} | 23.7 | 0.3 | 12.7 |
| rx2_distant | 0 | 0 | 0 | 0 |  |  |  |  |
| meter_rx2 | 12 | 0 | 2 | 10 |  | 22.7 | 0.1 | 16.0 |
| meter_on_rx | 2 | 1 | 1 | 0 | {'correct_parameters': 1} | 22.5 | 3.0 | 12.0 |
| control_rx | 0 | 0 | 0 | 0 |  |  |  |  |
| control_meter | 31 | 4 | 19 | 8 | {'correct_measurements': 4, 'get_measurement_context': 1} | 28.4 | 3.9 | 13.4 |
| all | 51 | 9 | 22 | 20 | {'correct_parameters': 5, 'correct_measurements': 4, 'get_measurement_context': 1} | 26.3 | 2.5 | 13.4 |

Max normalized residual after the first committed correction, alarm still on (40 actions), by committed correction -> next true correction:

| committed -> next true | n | min | p10 | median | p90 | share >= 10 |
| --- | --- | --- | --- | --- | --- | --- |
| correct_parameters -> correct_parameters | 287 | 2.6 | 4.9 | 9.5 | 23.7 | 48% |
| correct_parameters -> correct_measurements | 64 | 8.1 | 9.9 | 17.3 | 25.9 | 89% |
| correct_measurements -> correct_measurements | 39 | 6.9 | 11.2 | 15.5 | 23.2 | 97% |
| correct_parameters -> none | 37 | 2.5 | 4.4 | 7.2 | 31.6 | 35% |
| correct_measurements -> correct_parameters | 36 | 3.2 | 4.6 | 7.4 | 11.4 | 17% |
| correct_topology -> none | 20 | 6.9 | 11.6 | 19.5 | 29.5 | 90% |
| correct_measurements -> none | 12 | 5.0 | 5.1 | 7.5 | 15.9 | 33% |
| correct_topology -> correct_parameters | 3 | 4.7 | 4.7 | 27.3 | 32.6 | 67% |
| correct_topology -> correct_measurements | 3 | 11.9 | 11.9 | 21.4 | 25.8 | 100% |

