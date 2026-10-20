### correct_parameters: 96 decisions, 219 candidates (179 successful)
- **logistic** (AUC 0.685): rule 92 / ranker 91 / oracle 96 of 96 decisions; ranker gained 0, lost 1
  | stratum | decisions | rule | ranker | oracle | gained | lost |
  | --- | --- | --- | --- | --- | --- | --- |
  | measurement+parameter:ambiguous | 32 | 32 | 32 | 32 | 0 | 0 |
  | measurement+parameter:dominant | 8 | 8 | 7 | 8 | 0 | 1 |
  | measurement+parameter:misranked | 4 | 1 | 1 | 4 | 0 | 0 |
  | parameter:ambiguous | 40 | 40 | 40 | 40 | 0 | 0 |
  | parameter:dominant | 8 | 8 | 8 | 8 | 0 | 0 |
  | parameter:misranked | 4 | 3 | 3 | 4 | 0 | 0 |
- **gbm** (AUC 0.822): rule 92 / ranker 89 / oracle 96 of 96 decisions; ranker gained 0, lost 3
  | stratum | decisions | rule | ranker | oracle | gained | lost |
  | --- | --- | --- | --- | --- | --- | --- |
  | measurement+parameter:ambiguous | 32 | 32 | 32 | 32 | 0 | 0 |
  | measurement+parameter:dominant | 8 | 8 | 7 | 8 | 0 | 1 |
  | measurement+parameter:misranked | 4 | 1 | 1 | 4 | 0 | 0 |
  | parameter:ambiguous | 40 | 40 | 39 | 40 | 0 | 1 |
  | parameter:dominant | 8 | 8 | 8 | 8 | 0 | 0 |
  | parameter:misranked | 4 | 3 | 2 | 4 | 0 | 1 |
### correct_measurements: 202 decisions, 678 candidates (582 successful)
- **logistic** (AUC 0.614): rule 181 / ranker 181 / oracle 189 of 202 decisions; ranker gained 0, lost 0
  | stratum | decisions | rule | ranker | oracle | gained | lost |
  | --- | --- | --- | --- | --- | --- | --- |
  | measurement+parameter:ambiguous | 32 | 32 | 32 | 32 | 0 | 0 |
  | measurement+parameter:dominant | 8 | 8 | 8 | 8 | 0 | 0 |
  | measurement+parameter:misranked | 4 | 1 | 1 | 1 | 0 | 0 |
  | mimic:flow_pair_opposite_sign | 42 | 33 | 33 | 38 | 0 | 0 |
  | mimic:flow_pair_same_sign | 43 | 39 | 39 | 40 | 0 | 0 |
  | multi_measurement | 72 | 68 | 68 | 70 | 0 | 0 |
  | parameter:misranked | 1 | 0 | 0 | 0 | 0 | 0 |
- **gbm** (AUC 0.619): rule 181 / ranker 183 / oracle 189 of 202 decisions; ranker gained 2, lost 0
  | stratum | decisions | rule | ranker | oracle | gained | lost |
  | --- | --- | --- | --- | --- | --- | --- |
  | measurement+parameter:ambiguous | 32 | 32 | 32 | 32 | 0 | 0 |
  | measurement+parameter:dominant | 8 | 8 | 8 | 8 | 0 | 0 |
  | measurement+parameter:misranked | 4 | 1 | 1 | 1 | 0 | 0 |
  | mimic:flow_pair_opposite_sign | 42 | 33 | 35 | 38 | 2 | 0 |
  | mimic:flow_pair_same_sign | 43 | 39 | 39 | 40 | 0 | 0 |
  | multi_measurement | 72 | 68 | 68 | 70 | 0 | 0 |
  | parameter:misranked | 1 | 0 | 0 | 0 | 0 | 0 |
