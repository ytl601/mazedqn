# 补充核心导入（关键修复）
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

# ====================== 1. 优先经验回放（PER） ======================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, device="cpu"):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = torch.zeros((capacity,), dtype=torch.float32)
        self.pos = 0
        self.device = device

    def add(self, experience):
        max_prio = self.priorities[:len(self.buffer)].max().item() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        
        probs = prios ** self.alpha
        probs = probs / probs.sum()
        
        indices = torch.multinomial(probs, batch_size, replacement=False)
        samples = [self.buffer[idx.item()] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = weights.to(self.device)
        
        states = torch.tensor([s[0] for s in samples], dtype=torch.float32).to(self.device)
        actions = torch.tensor([s[1] for s in samples], dtype=torch.long).to(self.device)
        rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([s[3] for s in samples], dtype=torch.float32).to(self.device)
        dones = torch.tensor([s[4] for s in samples], dtype=torch.float32).to(self.device)
        
        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx.item()] = prio

    def __len__(self):
        return len(self.buffer)

# ====================== 2. DQN模型 ======================
class DQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=4, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.clamp_min = -20.0
        self.clamp_max = 20.0

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.clamp(x, self.clamp_min, self.clamp_max)

# ====================== 3. 迷宫环境 ======================
class WalkerEnv:
    def __init__(self, grid_size=5, target_pos=(4, 4), start_pos=(0, 0), render=False):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.current_pos = start_pos
        self.step_count = 0
        self.max_steps = 80
        self.action_space = 4  # 0:上,1:下,2:左,3:右
        self.action_map = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        self.state_dim = 4
        self.obstacles = {(1,1), (2,1), (4,1), (1,2), (4,2), (1,3)}
        
        # 渲染配置
        self.render_flag = render
        self.pygame = None
        if self.render_flag:
            try:
                import pygame
                self.pygame = pygame
                self.pygame.init()
                self.screen_size = 400
                self.cell_size = self.screen_size // self.grid_size
                self.screen = self.pygame.display.set_mode((self.screen_size, self.screen_size))
                self.pygame.display.set_caption("5x5 Maze DQN - Validation")
                self.clock = self.pygame.time.Clock()
            except ImportError:
                print("⚠️ Pygame未安装，禁用渲染")
                self.render_flag = False
        
        # 轨迹跟踪
        self.visited = set()
        self.last_pos = None
        self.repeat_count = 0
        self.min_dist_to_target = self._calc_dist(self.current_pos, self.target_pos)

    def _calc_dist(self, pos1, pos2):
        """曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self):
        self.current_pos = self.start_pos
        self.step_count = 0
        self.visited = {self.start_pos}
        self.last_pos = None
        self.repeat_count = 0
        self.min_dist_to_target = self._calc_dist(self.current_pos, self.target_pos)
        return self._get_state()

    def _get_state(self):
        x, y = self.current_pos
        tx, ty = self.target_pos
        
        # 归一化坐标
        norm_x = x / (self.grid_size - 1)
        norm_y = y / (self.grid_size - 1)
        
        # 终点方向（曼哈顿距离归一化）
        dx = tx - x
        dy = ty - y
        dist = self._calc_dist((x,y), (tx,ty)) + 1e-6
        dir_x = dx / dist
        
        # 最近障碍距离（归一化）
        min_obs_dist = self.grid_size
        for (ox, oy) in self.obstacles:
            obs_dist = self._calc_dist((x,y), (ox,oy))
            if obs_dist < min_obs_dist:
                min_obs_dist = obs_dist
        norm_obs_dist = min_obs_dist / self.grid_size
        
        return [norm_x, norm_y, dir_x, norm_obs_dist]

    def step(self, action):
        self.step_count += 1
        x, y = self.current_pos
        dx, dy = self.action_map[action]
        new_x = x + dx
        new_y = y + dy
        done = False
        info = {}

        # 边界/障碍检测
        valid_move = True
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            valid_move = False
        if (new_x, new_y) in self.obstacles:
            valid_move = False

        # 更新位置
        old_pos = self.current_pos
        if valid_move:
            self.current_pos = (new_x, new_y)
        else:
            self.current_pos = (x, y)

        # 奖励系统
        reward = 0.0
        reward -= 0.1  # 步数惩罚
        if not valid_move:
            reward -= 2.0  # 撞墙重罚
            self.repeat_count += 1
        
        # 重复访问惩罚
        if self.current_pos in self.visited:
            reward -= 1.5
            self.repeat_count += 1
        else:
            reward += 0.5
            self.visited.add(self.current_pos)
            self.repeat_count = 0
        
        # 进度奖励
        current_dist = self._calc_dist(self.current_pos, self.target_pos)
        if current_dist < self.min_dist_to_target:
            reward += 3.0
            self.min_dist_to_target = current_dist
        elif current_dist > self.min_dist_to_target:
            reward -= 1.0
        
        # 终点奖励
        if self.current_pos == self.target_pos:
            reward += 50.0
            done = True
            info["success"] = True
        
        # 超时/卡壳惩罚
        if self.step_count >= self.max_steps or self.repeat_count > 5:
            done = True
            info["success"] = False
            reward -= 10.0

        return self._get_state(), reward, done, info

    def get_action_mask(self):
        """动作掩码：1=有效，0=无效"""
        x, y = self.current_pos
        mask = [1]*4
        for action in range(4):
            dx, dy = self.action_map[action]
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                mask[action] = 0
            if (nx, ny) in self.obstacles:
                mask[action] = 0
        return mask

    def render(self):
        if not self.render_flag or self.pygame is None:
            return
        
        # 处理退出事件
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()
                exit()
        
        # 绘制界面
        self.screen.fill((255, 255, 255))
        for i in range(self.grid_size + 1):
            self.pygame.draw.line(self.screen, (0,0,0), (i*self.cell_size, 0), (i*self.cell_size, self.screen_size), 1)
            self.pygame.draw.line(self.screen, (0,0,0), (0, i*self.cell_size), (self.screen_size, i*self.cell_size), 1)
        
        # 绘制障碍
        for (x, y) in self.obstacles:
            self.pygame.draw.rect(self.screen, (100, 100, 100), (x*self.cell_size + 2, y*self.cell_size + 2, self.cell_size - 4, self.cell_size - 4))
        
        # 绘制起点/终点
        self.pygame.draw.rect(self.screen, (0, 255, 0), (self.start_pos[0]*self.cell_size + 2, self.start_pos[1]*self.cell_size + 2, self.cell_size - 4, self.cell_size - 4))
        self.pygame.draw.rect(self.screen, (255, 0, 0), (self.target_pos[0]*self.cell_size + 2, self.target_pos[1]*self.cell_size + 2, self.cell_size - 4, self.cell_size - 4))
        
        # 绘制小球
        cx = self.current_pos[0] * self.cell_size + self.cell_size // 2
        cy = self.current_pos[1] * self.cell_size + self.cell_size // 2
        self.pygame.draw.circle(self.screen, (0, 0, 255), (cx, cy), self.cell_size // 3)
        
        # 更新画面
        self.pygame.display.update()
        self.clock.tick(60)

    def close(self):
        if self.render_flag and self.pygame is not None:
            self.pygame.quit()

# ====================== 4. DQN智能体 ======================
class DQNAgent:
    def __init__(self):
        self.env = WalkerEnv(render=True)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_space
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前训练设备：{self.device}")

        # 模型初始化
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        # 优先经验回放
        self.memory = PrioritizedReplayBuffer(capacity=20000, alpha=0.6, beta=0.4, device=self.device)
        self.batch_size = 64

        # 探索策略
        self.epsilon_start = 0.8
        self.epsilon_end = 0.05
        self.epsilon_decay = 500
        self.steps_done = 0

        # 折扣因子
        self.gamma = 0.95

    def get_epsilon(self):
        """线性衰减探索率"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        return max(self.epsilon_end, epsilon)

    def choose_action(self, state):
        """纯贪心选择（验证时禁用探索）"""
        epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            mask = self.env.get_action_mask()
            valid_actions = [i for i, m in enumerate(mask) if m == 1]
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(state_tensor)
            
            # 动作掩码
            mask = torch.tensor(self.env.get_action_mask(), dtype=torch.float32).to(self.device)
            q_vals = q_vals * mask - 1e9 * (1 - mask)
            
            action = q_vals.argmax().item()
        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        # 采样经验
        (states, actions, rewards, next_states, dones), indices, weights = self.memory.sample(self.batch_size)

        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN目标Q值
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 计算损失
        loss = self.loss_fn(current_q, target_q)
        loss = (loss * weights).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新目标网络
        if self.steps_done % 50 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 更新优先级
        td_error = torch.abs(current_q - target_q).detach().cpu()
        self.memory.update_priorities(indices, td_error)

        # 学习率衰减
        self.lr_scheduler.step()

        return loss.item()

    def train(self, episodes=500):
        """训练智能体"""
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0.0
            loss_sum = 0.0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward

                # 存储经验
                self.memory.add((state, action, reward, next_state, done))

                # 学习
                loss = self.learn()
                loss_sum += loss

                # 更新状态
                state = next_state
                self.env.render()

            # 打印训练日志
            avg_loss = loss_sum / self.env.step_count if self.env.step_count > 0 else 0.0
            success = info.get("success", False)
            epsilon = self.get_epsilon()
            print(f"Train Episode {ep+1:4d} | Reward: {total_reward:6.1f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f} | Success: {success}")

        self.env.close()

# ====================== 5. 验证核心逻辑 ======================
def validate_agent(model_path=None, episodes=100, render=True):
    """
    验证训练后的智能体
    :param model_path: 训练好的模型路径
    :param episodes: 验证轮数
    :param render: 是否可视化
    """
    # 初始化环境和模型
    env = WalkerEnv(render=render)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(env.state_dim, env.action_space).to(device)
    
    # 加载训练好的模型
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ 成功加载模型：{model_path}")
        except Exception as e:
            print(f"❌ 加载模型失败：{e} | 使用随机初始化模型")
    else:
        print("⚠️ 未指定模型路径，使用随机初始化模型（仅作对比）")
    
    model.eval()  # 评估模式

    # 初始化验证指标
    total_success = 0
    total_steps = 0
    total_reward = 0
    success_episodes = []
    fail_episodes = []

    # 开始验证
    print("\n========== 开始验证 ==========")
    print(f"验证轮数：{episodes} | 渲染：{render} | 设备：{device}")
    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False
        path = [env.current_pos]

        while not done:
            # 纯贪心选择（禁用探索）
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_vals = model(state_tensor)
                mask = torch.tensor(env.get_action_mask(), dtype=torch.float32).to(device)
                q_vals = q_vals * mask - 1e9 * (1 - mask)
                action = q_vals.argmax().item()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            path.append(env.current_pos)
            
            # 可视化
            env.render()
            
            # 更新状态
            state = next_state

        # 统计指标
        total_success += 1 if info["success"] else 0
        total_steps += episode_steps
        total_reward += episode_reward

        if info["success"]:
            success_episodes.append(ep+1)
            print(f"Val Episode {ep+1:4d} | 成功 | 步数：{episode_steps:3d} | 奖励：{episode_reward:6.1f}")
        else:
            fail_episodes.append(ep+1)
            print(f"Val Episode {ep+1:4d} | 失败 | 步数：{episode_steps:3d} | 奖励：{episode_reward:6.1f} | 最后位置：{env.current_pos}")

    # 计算汇总指标
    avg_steps = total_steps / episodes
    avg_reward = total_reward / episodes
    success_rate = total_success / episodes * 100

    # 输出验证报告
    print("\n========== 验证报告 ==========")
    print(f"总验证轮数：{episodes}")
    print(f"成功轮数：{total_success} | 失败轮数：{episodes - total_success}")
    print(f"成功率：{success_rate:.2f}%")
    print(f"平均步数：{avg_steps:.2f} | 平均奖励：{avg_reward:.2f}")
    
    # 效果判断
    print("\n========== 效果判断 ==========")
    if success_rate >= 90:
        print("✅ 优秀：成功率≥90%，算法稳定收敛")
    elif success_rate >= 70:
        print("⚠️ 良好：成功率70%-90%，需少量调优")
    elif success_rate >= 50:
        print("⚠️ 一般：成功率50%-70%，需优化奖励/模型")
    else:
        print("❌ 较差：成功率<50%，需重构奖励系统")

    env.close()
    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward
    }

# ====================== 6. 训练+验证一体化 ======================
def train_and_validate(train_episodes=500, val_episodes=100, save_model_path="dqn_maze_model.pth"):
    """先训练，再验证，并保存模型"""
    # 1. 训练智能体
    agent = DQNAgent()
    agent.train(episodes=train_episodes)
    
    # 2. 保存模型
    torch.save(agent.policy_net.state_dict(), save_model_path)
    print(f"\n✅ 模型已保存至：{save_model_path}")
    
    # 3. 验证模型
    validate_agent(model_path=save_model_path, episodes=val_episodes, render=True)

# ====================== 执行入口 ======================
if __name__ == "__main__":
    # 自动安装依赖
    try:
        import pygame
    except ImportError:
        print("📦 正在安装pygame...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        import pygame

    # 选择执行方式：
    # 方式1：仅训练
    # agent = DQNAgent()
    # agent.train(episodes=500)

    # 方式2：仅验证（需先训练并保存模型）
    #validate_agent(model_path="dqn_maze_model.pth", episodes=100, render=True)

    # 方式3：训练+验证（推荐）
    train_and_validate(train_episodes=500, val_episodes=100)