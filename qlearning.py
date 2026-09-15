from envs.gridworld import Gridworld
from graph import read_layout, build_adjacency
import numpy as np

maze_path='envs/maze1.txt'
start, goal, obstacles, size = read_layout(maze_path)
print(start, goal, obstacles, size)
states, adj = build_adjacency(size, obstacles)

a = Gridworld(size=size, render_mode="human", obstacles=obstacles)
obs, info = a.reset(seed=42)



class EpsilonGreedy:
    def __init__(self, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, Q, state, action_mask):
        valid_actions = np.where(action_mask == 1)[0]

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)

        q_vals = Q[state].copy()
        q_vals[action_mask == 0] = -np.inf
        return np.argmax(q_vals)

    def update(self, state, action):
        pass  # no per-step bookkeeping needed

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class UCB:
    def __init__(self, n_states, n_actions, c=2.0):
        self.c = c
        self.counts = np.zeros((n_states, n_actions), dtype=np.int64)
        self.t = 0  # total steps taken, used in the log(t) term

    def select_action(self, Q, state, action_mask):
        self.t += 1
        valid_actions = np.where(action_mask == 1)[0]

        # any valid action never taken gets picked first (avoids log(0)/div0)
        unvisited = [a for a in valid_actions if self.counts[state, a] == 0]
        if unvisited:
            return np.random.choice(unvisited)

        ucb_vals = Q[state, valid_actions] + self.c * np.sqrt(
            np.log(self.t) / self.counts[state, valid_actions]
        )
        return valid_actions[np.argmax(ucb_vals)]

    def update(self, state, action):
        self.counts[state, action] += 1

    def end_episode(self):
        pass


class ClassicalWalk(EpsilonGreedy):
    def __init__(self, walk, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        super().__init__(epsilon, epsilon_min, epsilon_decay)
        self.walk = walk
    
    

def greedy(Q, state, action_mask):
    q_vals = Q[state].copy()
    q_vals[action_mask == 0] = -np.inf
    return np.argmax(q_vals)


def train_q_learning(policy, env, episodes=2000, alpha=0.1, gamma=0.95):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    for ep in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            action = policy.select_action(Q, state, info["action_mask"])
            policy.update(state, action)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next * (not terminated) - Q[state, action])

            state = next_state

        policy.end_episode()

        if (ep + 1) % 200 == 0:
            print(f"Episode {ep+1}")

    return Q, env


def evaluate(Q, env, episodes=5, render=True):
    env.render_mode = "human" if render else None
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = greedy(Q, state, info["action_mask"])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        print(f"Eval episode {ep+1}: reward={total_reward}, steps={steps}")


if __name__ == "__main__":
    n_states = 5 * 5
    n_actions = 4

    # --- switch here ---
    policy = EpsilonGreedy(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
    # policy = UCB(n_states, n_actions, c=2.0)

    Q, env = train_q_learning(policy, a)
    evaluate(Q, env)



"""
for _ in range(20):
    action = env.action_space.sample(mask=info["action_mask"])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print("Reached goal!")
        break
"""