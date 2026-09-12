from envs.gridworld import Gridworld
from graph import read_layout, build_adjacency

maze_path='envs/maze1.txt'
start, goal, obstacles, size = read_layout(maze_path)
print(start, goal, obstacles, size)

states, adj = build_adjacency(size, obstacles)


env = Gridworld(size=5, render_mode="human", obstacles=obstacles)
obs, info = env.reset(seed=42)

for _ in range(20):
    action = env.action_space.sample(mask=info["action_mask"])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print("Reached goal!")
        break
