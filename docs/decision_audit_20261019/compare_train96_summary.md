Variant `ledger_ranked_compare` against baseline `output/decision_audit_20261019/control_train96`.

| stratum | roots | solved | mean steps | roots compared | comparisons | best = true line | first line changed | baseline solved | baseline mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| measurement+parameter:ambiguous | 32 | 32 | 11.0 | 32 | 32 | 32/32 | 0 | 32 | 10.0 |
| measurement+parameter:dominant | 8 | 8 | 10.1 | 1 | 1 | 1/1 | 0 | 8 | 10.0 |
| measurement+parameter:misranked | 4 | 4 | 11.0 | 4 | 4 | 4/4 | 4 | 1 | 10.0 |
| parameter:ambiguous | 40 | 40 | 7.0 | 40 | 40 | 40/40 | 0 | 40 | 6.0 |
| parameter:dominant | 8 | 8 | 6.1 | 1 | 1 | 1/1 | 0 | 8 | 6.0 |
| parameter:misranked | 4 | 4 | 7.0 | 4 | 4 | 4/4 | 4 | 3 | 7.0 |
| all | 96 | 96 | 8.7 | 82 | 82 | 82/82 | 8 | 92 | 7.9 |

Roots whose outcome or step count (by 2 or more) changed:

- `r0_1e263d720ffb` (parameter:misranked, true line [6]): baseline failed in 10 -> variant solved in 7; comparisons [{'line_indices': [3, 6], 'ranked_line_indices': [6, 3], 'best_line_index': 6}]; first correction {'step': 3, 'line_index': 6, 'context_top_line': 3, 'dominant': False, 'dominance_ratio': 1.1822975038206827, 'execution_status': 'success'}
- `r0_96f831d958dc` (measurement+parameter:misranked, true line [6]): baseline failed in 10 -> variant solved in 11; comparisons [{'line_indices': [3, 6], 'ranked_line_indices': [6, 3], 'best_line_index': 6}]; first correction {'step': 7, 'line_index': 6, 'context_top_line': 3, 'dominant': False, 'dominance_ratio': 1.0710742083061018, 'execution_status': 'success'}
- `r0_a8ab061f6aa7` (measurement+parameter:misranked, true line [6]): baseline failed in 10 -> variant solved in 11; comparisons [{'line_indices': [3, 6], 'ranked_line_indices': [6, 3], 'best_line_index': 6}]; first correction {'step': 7, 'line_index': 6, 'context_top_line': 3, 'dominant': False, 'dominance_ratio': 1.0383499803724867, 'execution_status': 'success'}
- `r0_b8b0b2e6e5da` (measurement+parameter:misranked, true line [6]): baseline failed in 10 -> variant solved in 11; comparisons [{'line_indices': [3, 6, 4], 'ranked_line_indices': [6, 3, 4], 'best_line_index': 6}]; first correction {'step': 3, 'line_index': 6, 'context_top_line': 3, 'dominant': True, 'dominance_ratio': 1.319856182090114, 'execution_status': 'success'}
