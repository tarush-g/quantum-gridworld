import numpy as np
import walks, graph

def greedy(Q, state, action_mask):
    q_vals = Q[state].copy()
    q_vals[action_mask == 0] = -np.inf
    return np.argmax(q_vals)


def bonus1(env):
    P = walks.transition_matrix(env.adjacencies)
    pi = walks.szegedy(P, 50, env.start)


def train_q_learning(env, episodes=2000, alpha=0.1, gamma=0.95, beta=0.95):
    """
    alpha = learning rate
    gamma = discount rate
    beta = bonus scale rate
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    epsilon=1.0
    epsilon_min=0.05
    epsilon_decay=0.995
    Q = np.zeros((n_states, n_actions))
    total_steps=0

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        step_count=0

        while not done:
            valid_actions = np.where(info["action_mask"] == 1)[0]

            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                q_vals = Q[state].copy()
                q_vals[info["action_mask"] == 0] = -np.inf
                action = np.argmax(q_vals)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next * (not terminated) - Q[state, action])

            state = next_state
            step_count+=1
            total_steps+=1


        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {ep}, Steps: {step_count}")
        
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


