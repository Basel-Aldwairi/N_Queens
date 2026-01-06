import numpy as np
import copy
import random
import cv2


def count_attacking(board):
    N = len(board)
    count = 0
    attacking_queens = np.zeros(N)

    for col_1 in range(N - 1):
        row_1 = board[col_1]

        for j in range(1, N - col_1):
            col_2 = col_1 + j
            row_2 = board[col_2]

            if row_1 == row_2 or np.abs(row_1 - row_2) == np.abs(col_1 - col_2):
                attacking_queens[col_1] += 1
                attacking_queens[col_2] += 1
                count += 1

    return count, attacking_queens.astype(int)


def is_safe(board, col):
    row = board[col]
    for col_p in range(col):
        row_p = board[col_p]

        if row == row_p or np.abs(row - row_p) == np.abs(col - col_p):
            return False
    return True


def n_queens_backtracking(N=8):
    board = np.empty(N)
    expanded_nodes = 0

    def backtrack(board, col):
        nonlocal expanded_nodes

        for row in range(N):
            expanded_nodes += 1
            board[col] = row

            if is_safe(board, col):
                if col + 1 < N:
                    if backtrack(board, col + 1):
                        return True
                else:
                    return True

        return False

    return board.astype(int) if backtrack(board, 0) else None, expanded_nodes


def update_domains(domains: dict, col, row):
    domains[col] = {row}
    N = len(domains)
    for c in range(1, N - col):

        domain_c = copy.deepcopy(domains[col + c])

        r = row
        if row in domain_c:
            domain_c.remove(row)
        r = row + c
        if r in domain_c:
            domain_c.remove(r)
        r = row - c
        if r in domain_c:
            domain_c.remove(r)

        if domain_c == set():
            return False

        domains[col + c] = domain_c
        # print(domain_c)

    return domains


def domains_prune(domains, col, row, removed):
    N = len(domains)
    for c in range(col + 1, N):

        dist = c - col

        for r in (row, row + dist, row - dist):
            if r in domains[c]:
                domains[c].remove(r)
                removed.append((c, r))

        if not domains[c]:
            return False

    return True


def restore_domains(domains, removed):
    for col, row, in removed:
        domains[col].add(row)


def n_queens_backtracking_with_forwardchecking(N=8):
    board = np.full(N, -1)
    expanded_nodes = 0
    domains = {col: set(range(N)) for col in range(N)}

    def backtrack(col):
        nonlocal expanded_nodes

        if col == N:
            return True

        for row in list(domains[col]):

            expanded_nodes += 1
            board[col] = row

            original_domain = domains[col]
            domains[col] = {row}

            removed = []
            if domains_prune(domains, col, row, removed):
                if backtrack(col + 1):
                    return True

            restore_domains(domains, removed)
            domains[col] = original_domain
            board[col] = -1

        return False

    solved = backtrack(0)

    return board.astype(int) if solved else None, expanded_nodes


def n_queens_min_conflicts(N, max_depth=1000, max_restarts=100):
    expanded_nodes = 0

    for restart in range(max_restarts):
        board = []
        for i in range(N):
            board.append(np.random.randint(N))

        board = np.array(board)

        for i in range(max_depth):

            count, queens = count_attacking(board)
            if count == 0:
                return board, expanded_nodes

            conflicted_queens = np.where(queens > 0)[0]

            col = random.choice(conflicted_queens)

            heuristics = np.full(N, -1)
            original_row = board[col]
            for row in range(N):
                expanded_nodes += 1

                board[col] = row
                count, _ = count_attacking(board)
                heuristics[row] = count

            board[col] = original_row
            min_conflicts = heuristics.min()
            best_rows = np.where(heuristics == min_conflicts)[0]

            row = random.choice(best_rows)

            board[col] = row

    return False, expanded_nodes


def place_on_board(board_array, h=800, w=800):
    N = len(board_array)
    board_array = N - 1 - board_array

    board = np.zeros((h, w), dtype=np.uint8)

    board = np.zeros((h, w), dtype=np.uint8)

    box_h, box_w = h // N, w // N

    for row in range(N):
        for col in range(N):
            if (row + col) % 2 == 0:
                board[box_h * row:box_h * (row + 1), box_w * col:box_w * (col + 1)] = 128

    queen_path = '../data/queen.png'
    queen = cv2.imread(queen_path)
    queen = cv2.cvtColor(queen, cv2.COLOR_BGR2GRAY)
    _, queen = cv2.threshold(queen, 128, 255, cv2.THRESH_BINARY)
    queen = cv2.resize(queen, (box_h, box_w))

    mask = queen < 255

    placed_board = board.copy()

    for col, row in enumerate(board_array):
        placed_board[box_h * row:box_h * (row + 1), box_w * col:box_w * (col + 1)][mask] = queen[mask]

    _, board_preprocessed = cv2.threshold(placed_board, 0, 255, cv2.THRESH_BINARY)

    return board_preprocessed