import gym
from gym import spaces
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = {(1,1), (2,1), (4,1),
                          (1,2), (4,2), (1,3)}

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.size - 1, shape=(2,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.agent_pos = list(self.start)
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):
        x, y = self.agent_pos

        if action == 0: y -= 1
        elif action == 1: y += 1
        elif action == 2: x -= 1
        elif action == 3: x += 1

        if not (0 <= x < self.size and 0 <= y < self.size):
            x, y = self.agent_pos

        if (x, y) in self.obstacles:
            x, y = self.agent_pos

        self.agent_pos = [x, y]

        reward = -1
        done = False

        if (x, y) == self.goal:
            reward = 50
            done = True

        return np.array(self.agent_pos, dtype=np.float32), reward, done, {}

    def render(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        for ox, oy in self.obstacles:
            grid[oy][ox] = '#'

        gx, gy = self.goal
        grid[gy][gx] = 'G'

        ax, ay = self.agent_pos
        grid[ay][ax] = 'A'

        for row in grid:
            print(' '.join(row))
        print()
