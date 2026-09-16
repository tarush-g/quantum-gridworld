import numpy as np
import gymnasium as gym
from gymnasium import spaces

moves = [
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1)    # right
]

class Gridworld(gym.Env):
    def __init__(
        self,
        size=5,
        obstacles=None,
        render_mode=None,
        start_pos=(0,0),
        goal_pos=None,
        max_steps=None
    ):
        
        self.size = size
        self.start = start_pos
        self.goal = goal_pos if goal_pos is not None else (size-1, size-1)
        self.obstacles = obstacles if obstacles is not None else set()
        self.max_steps = max_steps if max_steps is not None else (4*size*size)

        self.states, self.adjacencies, self.successors = build_adjacency(self.size, self.obstacles)

        self.observation_space = spaces.Discrete(len(self.states))
        self.action_space = spaces.Discrete(len(moves))
        self.render_mode = render_mode
        self.agent_pos = None

    def _to_index(self, pos):
        return self.states[pos]

    def _to_coords(self, index):
        return divmod(index, self.size)

    def _get_obs(self):
        return self._to_index(self.agent_pos)

    def _get_info(self):
        agent = np.array(self.agent_pos)
        return {
            "distance": np.linalg.norm(agent - np.array(self.goal), ord=1),
            "action_mask": self._action_masks(),
        }

    def _action_masks(self):
        r, c = self.agent_pos
        masks = np.zeros(4, dtype=np.int8)
        for action, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < self.size
                and 0 <= nc < self.size
                and (nr, nc) not in self.obstacles
            ):
                masks[action] = 1
        return masks

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.steps_taken = 0
        self.agent_pos = self.start

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        r, c = self.agent_pos
        if action == 0:   r -= 1   # up
        elif action == 1: r += 1   # down
        elif action == 2: c -= 1   # left
        elif action == 3: c += 1   # right

        r = max(0, min(self.size - 1, r))
        c = max(0, min(self.size - 1, c))

        if (r, c) not in self.obstacles:
            self.agent_pos = (r, c)

        self.steps_taken += 1
        terminated = self.agent_pos == self.goal
        reward = 1.0 if terminated else 0.0
        truncated = self.steps_taken >= self.max_steps and not terminated

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        grid = np.zeros((self.size, self.size), dtype=np.int8)
        for (r, c) in self.obstacles:
            grid[r, c] = 1
        gr, gc = self.goal
        grid[gr, gc] = 3
        ar, ac = self.agent_pos
        grid[ar, ac] = 2

        # . = empty, # = obstacle, A = agent, G = goal
        symbols = {0: ".", 1: "#", 2: "A", 3: "G"}
        for row in grid:
            print(" ".join(symbols[cell] for cell in row))
        print()


def build_adjacency(size, obstacles):
    """
    states = index map dict with {(row, col): state_index}
    adjacencies = adjacency matrix of the grid
    successors = array where successors[state_index][action] = next state_index
    """
    obstacles = set(obstacles or [])
    states = {}

    for row in range(size):
        for col in range(size):
            if (row, col) not in obstacles:
                states[(row, col)] = len(states)

    # Create adjacency matrix
    num_states = len(states)
    adjacencies = np.zeros((num_states, num_states), dtype=np.int8)
    successors = np.zeros((num_states, len(moves)), dtype=int)

    # Add connections
    for (row, col), index in states.items():
        for action, (dr, dc) in enumerate(moves):
            neighbor = (row + dr, col + dc)
            if neighbor in states:
                adjacencies[index, states[neighbor]] = 1
                successors[index, action] = states[neighbor]
            else:
                successors[index, action] = index

    return states, adjacencies, successors
