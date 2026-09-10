import numpy as np
from collections import deque


moves = [
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1)    # right
]

def build_adjacency(size, obstacles):
    obstacles = set(obstacles or [])
    states = {}

    for row in range(size):
        for col in range(size):
            if (row, col) not in obstacles:
                states[(row, col)] = len(states)

    # Create adjacency matrix
    num_states = len(states)
    adjacencies = np.zeros((num_states, num_states), dtype=np.int8)

    # Add connections
    for (row, col), index in states.items():
        for dr, dc in moves:
            neighbor = (row + dr, col + dc)
            if neighbor in states:
                adjacencies[index, states[neighbor]] = 1

    return states, adjacencies

def is_reachable(start, goal, obstacles, size):
    if start == goal:
        return True

    visited = {start}
    queue = deque([start])

    while queue:
        row, col = queue.popleft()
        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            neighbor = (nr, nc)

            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if neighbor in obstacles or neighbor in visited:
                continue
            if neighbor == goal:
                return True
            
            visited.add(neighbor)
            queue.append(neighbor)

    return False


def read_layout(path):
    obstacles = set()
    try:
        with open(path, "r") as f:
            rows = ["".join(line.split()) for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found")
    except OSError:
        raise OSError(f"Could not read file")

    # Ignore fully blank lines
    rows = [r for r in rows if r != ""]

    if not rows:
        raise ValueError(f"File is empty")

    for row, line in enumerate(rows):
        for col, cell in enumerate(line):
            if cell == "#":
                obstacles.add((row, col))
            elif cell == "A":
                start = (row, col)
            elif cell == "G":
                goal = (row, col)

    size = len(rows[0])
    height = len(rows)
    if height != size:
        raise ValueError(f"Must be square: {height} rows but {size} columns")

    if not is_reachable(start, goal, obstacles, size):
        raise ValueError(f"Goal not reachable")
    
    return start, goal, obstacles, size


"""Row-normalise adjacency matrix into a Markov chain transition matrix."""
def transition_matrix(adj):
    adj = np.asarray(adj, dtype=float)
    row_sums = adj.sum(axis=1, keepdims=True)
    P = adj / row_sums
    return P
