# Imports
import numpy as np
import copy
import random
import cv2
from pathlib import Path

# Check if the queen placeced doesn't break conditions
def is_safe(board, col):
    row = board[col]
    for col_p in range(col):
        row_p = board[col_p]

        if row == row_p or np.abs(row - row_p) == np.abs(col - col_p):
            return False
    return True


# Backtracking algorithm
def n_queens_backtracking(N=8):
    # Starts with an empty board
    board = np.full(N,-1)
    # Tracks number of nodes expanded
    expanded_nodes = 0

    # Backtracking
    def backtrack(board, col):
        nonlocal expanded_nodes

        # Check every row for the coulumn of the recursion
        for row in range(N):
            expanded_nodes += 1
            board[col] = row

            if is_safe(board, col):
                # If queen placement is safe,
                if col + 1 < N:
                    # but not the last queen,
                    # continue to nex Queen
                    if backtrack(board, col + 1):
                        return True
                else:
                    # Last queen is safe, propogate True
                    return True

        # No safe states found, N= 2, 3
        return False

    # Return the solved board, and the number of expanded nodes
    return board.astype(int) if backtrack(board, 0) else None, expanded_nodes


# Update the domain in backtracking with forward checking,
# after placing a specific queen on a specific row
def domains_prune(domains, col, row, removed):
    N = len(domains)

    for c in range(col + 1, N):
        # Iterate over each queen, remove impossible domains
        dist = c - col

        for r in (row, row + dist, row - dist):
            if r in domains[c]:
                domains[c].remove(r)
                # Add removed possible rows to removed,
                # so we can bring the back in case of failure
                removed.append((c, r))

        # If empty domain for any queen, failure
        if not domains[c]:
            return False

    # Update successful
    return True

# Bring back original domain, if domais_prune fails
def restore_domains(domains, removed):
    for col, row, in removed:
        domains[col].add(row)

# Backtracking with forward checking
def n_queens_backtracking_with_forwardchecking(N=8):
    # Start with empty board
    board = np.full(N, -1)
    # Keeps track of number of expanded nodes
    expanded_nodes = 0
    # Keeps track of possible domains for each queen
    domains = {col: set(range(N)) for col in range(N)}

    # backtracking over a queen
    def backtrack(col):
        nonlocal expanded_nodes

        # If last queen is placed without error, backpropogate True
        if col == N:
            return True

        # Try every row the queen can be at from her domain
        for row in list(domains[col]):

            expanded_nodes += 1
            board[col] = row

            original_domain = domains[col]
            domains[col] = {row}

            # Check if placement satisfies restrictions
            removed = []
            if domains_prune(domains, col, row, removed):
                if backtrack(col + 1):
                    return True

            # If placement fails, restore initial state
            restore_domains(domains, removed)
            domains[col] = original_domain
            board[col] = -1

        return False

    solved = backtrack(0)

    return board.astype(int) if solved else None, expanded_nodes


# Heuristic: Counts the number of attacking pairs of Queens
def count_attacking(board):

    N = len(board)
    count = 0
    # Tracks how many queens each queen attacks
    attacking_queens = np.zeros(N)

    # Iterate over each queen
    for col_1 in range(N - 1):
        row_1 = board[col_1]

        # Iterate over each consecutive queen,
        # check if it is attacked by queen 1
        for j in range(1, N - col_1):
            col_2 = col_1 + j
            row_2 = board[col_2]

            # If queen 2 is attacked by queen 1, incremeant
            if row_1 == row_2 or np.abs(row_1 - row_2) == np.abs(col_1 - col_2):
                attacking_queens[col_1] += 1
                attacking_queens[col_2] += 1
                count += 1

    return count, attacking_queens.astype(int)

# Min Conflicts
def n_queens_min_conflicts(N, max_depth=1000, max_restarts=100):
    # Keeps track of number of expanded nodes
    expanded_nodes = 0

    # Random restart, to escape local minima
    for restart in range(max_restarts):

        # Starts with a random board state
        board = []
        for i in range(N):
            board.append(np.random.randint(N))
        board = np.array(board)

        # Iterative method, no recursion
        for i in range(max_depth):

            # Checks if solution is found
            count, queens = count_attacking(board)
            if count == 0:
                return board, expanded_nodes

            # If solution is not found,
            # pick a random queen in conflict
            conflicted_queens = np.where(queens > 0)[0]
            col = random.choice(conflicted_queens)

            # Heuristic, number of attacking pairs per node
            heuristics = np.full(N, -1)
            original_row = board[col]
            # Try every row for picked queen
            for row in range(N):
                expanded_nodes += 1

                board[col] = row
                count, _ = count_attacking(board)
                heuristics[row] = count

            # Pick row for picked queen, which results in
            # Lowest number of attacking pairs
            # Randomly pick between smallest, to avoid local minima problem
            board[col] = original_row
            min_conflicts = heuristics.min()
            best_rows = np.where(heuristics == min_conflicts)[0]

            row = random.choice(best_rows)

            board[col] = row

    return False, expanded_nodes


# Draw board with numpy, cv2, and PIl.Image
def draw_board(board_array, h=800, w=800):
    N = len(board_array)
    # Flip board vertically, for consitency
    board_array = N - 1 - board_array

    # Starts with a black image
    board = np.zeros((h, w), dtype=np.uint8)

    # Find height and widht for each box
    box_h, box_w = h // N, w // N

    # Place white boxes
    for row in range(N):
        for col in range(N):
            # Gives a checkerboard pattern
            if (row + col) % 2 == 0:
                board[box_h * row:box_h * (row + 1), box_w * col:box_w * (col + 1)] = 158

    # Find queen image file
    Root = Path(__file__).parent.parent
    queen_path = Root / 'data' / 'queen.png'
    queen = cv2.imread(queen_path)

    # Convert queen to gray scale
    queen = cv2.cvtColor(queen, cv2.COLOR_BGR2GRAY)
    # Resize queen to be same as a box
    queen = cv2.resize(queen, (box_h, box_w))
    # Threshold
    _, queen = cv2.threshold(queen, 128, 255, cv2.THRESH_BINARY)
    # Invert queen, so it becomes pure white
    queen = cv2.bitwise_not(queen)


    placed_board = board.copy()
    # Place queen depending on the board given
    for col, row in enumerate(board_array):
        board_location = board[box_h * row:box_h * (row + 1), box_w * col:box_w * (col + 1)]
        placed_board[box_h * row:box_h * (row + 1), box_w * col:box_w * (col + 1)] = cv2.bitwise_or(board_location,
                                                                                                     queen)

    return placed_board