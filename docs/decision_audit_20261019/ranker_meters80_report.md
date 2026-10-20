### correct_parameters: 0 decisions, 0 candidates (0 successful)
too few decisions or no label variation
### correct_measurements: 157 decisions, 521 candidates (446 successful)
- **logistic** (AUC 0.760): rule 140 / ranker 144 / oracle 148 of 157 decisions; ranker gained 5, lost 1
  | stratum | decisions | rule | ranker | oracle | gained | lost |
  | --- | --- | --- | --- | --- | --- | --- |
  | mimic:flow_pair_opposite_sign | 42 | 33 | 37 | 38 | 4 | 0 |
  | mimic:flow_pair_same_sign | 43 | 39 | 38 | 40 | 0 | 1 |
  | multi_measurement | 72 | 68 | 69 | 70 | 1 | 0 |
- **gbm** (AUC 0.710): rule 140 / ranker 142 / oracle 148 of 157 decisions; ranker gained 3, lost 1
  | stratum | decisions | rule | ranker | oracle | gained | lost |
  | --- | --- | --- | --- | --- | --- | --- |
  | mimic:flow_pair_opposite_sign | 42 | 33 | 36 | 38 | 3 | 0 |
  | mimic:flow_pair_same_sign | 43 | 39 | 38 | 40 | 0 | 1 |
  | multi_measurement | 72 | 68 | 68 | 70 | 0 | 0 |
