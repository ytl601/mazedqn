import pygame
import sys
import time

class MazeRenderer:
    def __init__(self, env, cell_size=100):
        pygame.init()

        self.env = env
        self.cell_size = cell_size
        self.size = env.size

        self.width = self.size * cell_size
        self.height = self.size * cell_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("DQN Maze")

        # 颜色
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED   = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE  = (0, 0, 255)

    def draw(self):
        self.screen.fill(self.WHITE)

        # 网格
        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)

        # 障碍物
        for ox, oy in self.env.obstacles:
            rect = pygame.Rect(
                ox * self.cell_size,
                oy * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, self.BLACK, rect)

        # 目标
        gx, gy = self.env.goal
        rect = pygame.Rect(
            gx * self.cell_size,
            gy * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.GREEN, rect)

        # 智能体
        ax, ay = self.env.agent_pos
        center = (
            ax * self.cell_size + self.cell_size // 2,
            ay * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(self.screen, self.RED, center, self.cell_size // 3)

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def tick(self, fps=10):
        self.handle_events()
        time.sleep(1 / fps)
