import numpy as np

MOVES = [
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1)    # right
]

def build_adjacency(size, obstacles):
    obstacles = set(obstacles or [])

    # Dict of valid states
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
        for dr, dc in MOVES:
            neighbor = (row + dr, col + dc)
            if neighbor in states:
                adjacencies[index, states[neighbor]] = 1

    return states, adjacencies

