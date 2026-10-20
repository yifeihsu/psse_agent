Variant `ledger_ranked_compare` against baseline `output/decision_audit_20261019/control_dev`.

| stratum | roots | solved | mean steps | roots compared | comparisons | best = true line | first line changed | baseline solved | baseline mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| measurement+parameter:ambiguous | 16 | 16 | 11.0 | 16 | 16 | 16/16 | 0 | 16 | 10.0 |
| measurement+parameter:dominant | 8 | 8 | 10.6 | 1 | 1 | 1/1 | 0 | 8 | 10.5 |
| measurement+parameter:misranked | 8 | 8 | 11.0 | 8 | 8 | 8/8 | 8 | 4 | 10.8 |
| parameter:ambiguous | 24 | 24 | 7.0 | 24 | 24 | 24/24 | 0 | 24 | 6.0 |
| parameter:dominant | 8 | 8 | 6.2 | 2 | 2 | 2/2 | 0 | 8 | 6.0 |
| parameter:misranked | 9 | 9 | 7.0 | 9 | 9 | 9/9 | 9 | 5 | 6.4 |
| all | 73 | 73 | 8.6 | 60 | 60 | 60/60 | 17 | 65 | 7.9 |

Roots whose outcome or step count (by 2 or more) changed:

- `r0_2f7496b3ef2c` (parameter:misranked, true line [6]): baseline failed in 10 -> variant solved in 7; comparisons [{'line_indices': [3, 6], 'ranked_line_indices': [6, 3], 'best_line_index': 6}]; first correction {'step': 3, 'line_index': 6, 'context_top_line': 3, 'dominant': False, 'dominance_ratio': 1.1822975038206827, 'execution_status': 'success'}
- `r0_3ad37c833dc5` (measurement+parameter:misranked, true line [6]): baseline failed in 10 -> variant solved in 11; comparisons [{'line_indices': [3, 6, 4], 'ranked_line_indices': [6, 3, 4], 'best_line_index': 6}]; first correction {'step': 7, 'line_index': 6, 'context_top_line': 3, 'dominant': True, 'dominance_ratio': 1.3036151181225657, 'execution_status': 'success'}
- `r0_4feaa499d7b8` (measurement+parameter:misranked, true line [6]): baseline failed in 10 -> variant solved in 11; comparisons [{'line_indices': [3, 6], 'ranked_line_indices': [6, 3], 'best_line_index': 6}]; first correction {'step': 7, 'line_index': 6, 'context_top_line': 3, 'dominant': False, 'dominance_ratio': 1.0710731526178983, 'execution_status': 'success'}
- `r0_8013605195d2` (parameter:misranked, true line [6]): baseline failed in 6 -> variant solved in 7; comparisons [{'line_indices': [3, 6, 4], 'ranked_line_indices': [6, 3, 4], 'best_line_index': 6}]; first correction {'step': 3, 'line_index': 6, 'context_top_line': 3, 'dominant': True, 'dominance_ratio': 1.303707550312555, 'execution_status': 'success'}
- `r0_9710751295b8` (parameter:misranked, true line [6]): baseline failed in 6 -> variant solved in 7; comparisons [{'line_indices': [3, 6], 'ranked_line_indices': [6, 3], 'best_line_index': 6}]; first correction {'step': 3, 'line_index': 6, 'context_top_line': 3, 'dominant': False, 'dominance_ratio': 1.071084274711645, 'execution_status': 'success'}
- `r0_bb0177c569e5` (parameter:misranked, true line [6]): baseline failed in 6 -> variant solved in 7; comparisons [{'line_indices': [3, 6], 'ranked_line_indices': [6, 3], 'best_line_index': 6}]; first correction {'step': 3, 'line_index': 6, 'context_top_line': 3, 'dominant': False, 'dominance_ratio': 1.0399851461898884, 'execution_status': 'success'}
- `r0_e5ec58bf6605` (measurement+parameter:misranked, true line [6]): baseline failed in 10 -> variant solved in 11; comparisons [{'line_indices': [3, 6], 'ranked_line_indices': [6, 3], 'best_line_index': 6}]; first correction {'step': 7, 'line_index': 6, 'context_top_line': 3, 'dominant': False, 'dominance_ratio': 1.040168278424302, 'execution_status': 'success'}
- `r0_e8b8de144865` (measurement+parameter:misranked, true line [6]): baseline failed in 16 -> variant solved in 11; comparisons [{'line_indices': [3, 6], 'ranked_line_indices': [6, 3], 'best_line_index': 6}]; first correction {'step': 7, 'line_index': 6, 'context_top_line': 3, 'dominant': False, 'dominance_ratio': 1.1821392489960862, 'execution_status': 'success'}
