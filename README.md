# N_Queens Problem

### Basel Al-Dwairi

This repo was the term project for the AI class at GJU first term 2025/2026

___

N_Queens is a classical AI problem. Where each row holds only
one queen. A satisfying result is reached when no queen 
attacks any other queen.

In this Repo, I explored ways to solve it using the following algorithms:


- Backtracking
- Backtracking with Forward Checking
- Min Conflicts

___

Main Features:
- Streamlit UI:
  - Choose the algorithm
    - `n_queens_backtracking`:
      - `is_safe`
    - `n_queens_backtracking_with_forwardchecking`:
      - `domains_prune`
      - `restore_domains`
    - `n_queens_min_conflicts`:
      - `count_attacking`
  - Choose N
  - View results and metrics
  - View rendered chess board with `draw_board`:
    - Function to render boards of N * N with N queens
  
- Modular Functions in `NQ_functions.py`
- 
___