from graph import read_layout, build_adjacency
from envs.gridworld import Gridworld
from qlearning import *


maze_path='envs/maze1.txt'
start, goal, obstacles, size = read_layout(maze_path)
print(start, goal, obstacles, size)

a = Gridworld(size=size, obstacles=obstacles)
obs, info = a.reset(seed=42)


if __name__ == "__main__":
    n_states = a.observation_space.n
    n_actions = 4
    
    Q, env = train_q_learning(a)
    evaluate(Q, env)



"""
for _ in range(20):
    action = env.action_space.sample(mask=info["action_mask"])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print("Reached goal!")
        break
"""