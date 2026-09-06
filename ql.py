import numpy as np
from gridworld import Gridworld
 
 
def train_q_learning(
    env,
    episodes=2000,
    alpha=0.1,       # learning rate
    gamma=0.95,      # discount factor
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    max_steps=200,
    seed=0,
):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
 
    rng = np.random.default_rng(seed)
    epsilon = epsilon_start
    episode_rewards = []
    episode_lengths = []
 
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
 
        for step in range(max_steps):
            mask = info["action_mask"]
            valid_actions = np.flatnonzero(mask)
 
            # epsilon-greedy over VALID actions only
            if rng.random() < epsilon:
                action = rng.choice(valid_actions)
            else:
                masked_q = np.where(mask.astype(bool), Q[obs], -np.inf)
                action = int(np.argmax(masked_q))
 
            next_obs, reward, terminated, truncated, next_info = env.step(action)
 
            # Q-learning update, bootstrapping only over valid next actions
            next_mask = next_info["action_mask"].astype(bool)
            if next_mask.any():
                best_next = np.max(np.where(next_mask, Q[next_obs], -np.inf))
            else:
                best_next = 0.0
            td_target = reward + gamma * best_next * (not terminated)
            Q[obs, action] += alpha * (td_target - Q[obs, action])
 
            obs, info = next_obs, next_info
            total_reward += reward
 
            if terminated or truncated:
                break
 
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
 
        if (ep + 1) % 200 == 0:
            avg_r = np.mean(episode_rewards[-200:])
            avg_len = np.mean(episode_lengths[-200:])
            print(
                f"Episode {ep + 1:5d} | avg reward (last 200): {avg_r:.3f} "
                f"| avg steps: {avg_len:6.1f} | epsilon: {epsilon:.3f}"
            )
 
    return Q, episode_rewards, episode_lengths
 
 
def run_greedy_episode(env, Q, max_steps=100, render=True):
    """Roll out the greedy policy learned in Q, optionally rendering it."""
    obs, info = env.reset(seed=123)
    total_reward = 0.0
 
    if render:
        env.render()
 
    for step in range(max_steps):
        mask = info["action_mask"].astype(bool)
        masked_q = np.where(mask, Q[obs], -np.inf)
        action = int(np.argmax(masked_q))
 
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
 
        if render:
            env.render()
 
        if terminated:
            print(f"Reached goal in {step + 1} steps! Total reward: {total_reward}")
            break
        if truncated:
            print("Truncated before reaching goal.")
            break
    else:
        print("Did not reach the goal within max_steps.")
 
    return total_reward
 
 
if __name__ == "__main__":
    train_env = Gridworld(size=5)
    Q, rewards, lengths = train_q_learning(train_env, episodes=2000)
 
    print("\nTraining complete. Learned Q-table shape:", Q.shape)
 
    print("\nGreedy rollout with rendering:")
    demo_env = Gridworld(size=5, render_mode="human")
    run_greedy_episode(demo_env, Q)
