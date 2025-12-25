from env_maze import MazeEnv
from dqn_agent import DQNAgent
from maze_render import MazeRenderer
import torch

env = MazeEnv()
agent = DQNAgent()

USE_RENDER = True     # ← 训练时开关
renderer = MazeRenderer(env) if USE_RENDER else None

episodes = 500

for ep in range(episodes):
    state = env.reset()

    for step in range(100):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.store(state, action, reward, next_state, done)
        agent.train()

        state = next_state

        if USE_RENDER and ep % 10 == 0:  # 每 10 个 episode 看一次
            renderer.draw()
            renderer.tick(fps=8)

        if done:
            break

    agent.update_target()
    print(f"Episode {ep}, epsilon={agent.epsilon:.3f}")

torch.save(agent.policy_net.state_dict(), "dqn_maze.pth")
print("已保存模型文件")
