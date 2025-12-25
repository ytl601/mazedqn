from env_maze import MazeEnv
from dqn_agent import DQNAgent
from maze_render import MazeRenderer
import torch

env = MazeEnv()
agent = DQNAgent()

# 加载训练好的模型
agent.policy_net.load_state_dict(torch.load("dqn_maze.pth"))
agent.epsilon = 0.0  # 纯利用

renderer = MazeRenderer(env)

state = env.reset()
done = False

while not done:
    action = agent.select_action(state)
    state, _, done, _ = env.step(action)

    renderer.draw()
    renderer.tick(fps=5)
