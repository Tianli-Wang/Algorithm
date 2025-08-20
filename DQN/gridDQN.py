import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# -------------------------------
# 1. 经验回放缓冲区
# -------------------------------
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# -------------------------------
# 2. DQN 网络结构（可扩展）
# -------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# -------------------------------
# 3. 网格环境（优化版）
# -------------------------------
class GridEnv:
    def __init__(self, size=10, num_obstacles=10, start=None, end=None):
        self.size = size
        self.num_obstacles = num_obstacles
        self.start = start or (0, 0)
        self.end = end or (size - 1, size - 1)

        # 确保起点 ≠ 终点
        while self.start == self.end:
            self.end = (random.randint(0, size - 1), random.randint(0, size - 1))

        # 随机生成障碍物（避开起点和终点）
        self.obstacles = set()
        while len(self.obstacles) < num_obstacles:
            obs = (random.randint(0, size - 1), random.randint(0, size - 1))
            if obs != self.start and obs != self.end:
                self.obstacles.add(obs)

        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.path = [self.start]
        return self._get_state()

    def _get_state(self):
        # 当前状态：[agent_x, agent_y, goal_x, goal_y]
        return np.array([*self.agent_pos, *self.end], dtype=np.float32)

    def step(self, action):
        x, y = self.agent_pos
        size = self.size

        # 动作定义：0=上, 1=右, 2=下, 3=左
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < size - 1:
            y += 1
        elif action == 2 and x < size - 1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1

        new_pos = (x, y)
        reward = -0.01  # 默认每步惩罚
        done = False

        if new_pos == self.end:
            reward = 10.0
            done = True
        elif new_pos in self.obstacles:
            reward = -5.0
            done = True
        elif new_pos in self.path[:-1]:  # 重复路径惩罚
            reward = -0.1

        self.agent_pos = new_pos
        self.path.append(new_pos)

        next_state = self._get_state()
        return next_state, reward, done

    def render(self, ax=None, show_path=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            show = True
        else:
            ax.clear()
            show = False

        grid = np.zeros((self.size, self.size))
        # 0: 空地, 0.3: 障碍物, 0.5: 起点, 0.7: 终点, 1.0: 智能体
        for ox, oy in self.obstacles:
            grid[ox, oy] = 0.3
        grid[self.start] = 0.5
        grid[self.end] = 0.7
        grid[self.agent_pos] = 1.0

        ax.imshow(grid, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
        if show_path and len(self.path) > 1:
            xs, ys = zip(*self.path)
            ax.plot(ys, xs, 'w--', linewidth=1.5, alpha=0.8)

        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(True, color='white', linewidth=0.5)
        ax.set_title(f"Step: {len(self.path)-1}, Pos: {self.agent_pos}")
        if show:
            plt.show()
        return ax


# -------------------------------
# 4. 动作选择（ε-greedy + decay）
# -------------------------------
def select_action(policy_net, state, n_actions, eps_threshold, device):
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# -------------------------------
# 5. 模型优化函数（优化版）
# -------------------------------
def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, device):
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # 构建张量
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device)

    # 当前Q值
    q_values = policy_net(state_batch).gather(1, action_batch)

    # 目标Q值
    next_state_values = torch.zeros(batch_size, device=device)
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_q = reward_batch + (gamma * next_state_values * (1 - done_batch))

    # 损失计算
    loss_fn = nn.SmoothL1Loss()  # Huber loss
    loss = loss_fn(q_values.squeeze(), expected_q)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # 更推荐使用 norm 裁剪
    optimizer.step()

    return loss.item()


# -------------------------------
# 6. 主训练函数（模块化 + 日志 + 保存）
# -------------------------------
def train_dqn(
    grid_size=10,
    num_obstacles=10,
    episodes=500,
    batch_size=64,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=800,
    tau=0.005,
    lr=1e-3,
    device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 环境与网络
    env = GridEnv(size=grid_size, num_obstacles=num_obstacles)
    n_states = 4
    n_actions = 4

    policy_net = DQN(n_states, 64, n_actions).to(device)
    target_net = DQN(n_states, 64, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    memory = ReplayMemory(10000)

    # 训练记录
    scores = []
    losses = []
    epsilons = []

    # 设置图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.ion()

    steps_done = 0
    best_score = -float('inf')

    for episode in range(episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        episode_loss = 0
        loss_count = 0

        # ε 衰减
        eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-steps_done / eps_decay)

        for t in range(100):  # 最大步数
            action = select_action(policy_net, state, n_actions, eps_threshold, device)
            steps_done += 1

            next_state_np, reward, done = env.step(action.item())
            total_reward += reward

            reward_tensor = torch.tensor([reward], device=device)
            next_state = None if done else torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward_tensor, done)
            state = next_state

            # 优化
            if len(memory) >= batch_size:
                loss = optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, device)
                if loss:
                    episode_loss += loss
                    loss_count += 1

            # 软更新 target network
            with torch.no_grad():
                for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if done:
                break

        # 记录
        scores.append(total_reward)
        epsilons.append(eps_threshold)
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        losses.append(avg_loss)

        # 保存最优模型
        if total_reward > best_score:
            best_score = total_reward
            torch.save(policy_net.state_dict(), "best_dqn_grid.pth")

        # 打印日志
        if episode % 50 == 0:
            print(f"Episode {episode:3d} | Score: {total_reward:5.2f} | Eps: {eps_threshold:4.2f} | Avg Loss: {avg_loss:6.4f}")

        # 实时可视化
        if episode % 10 == 0:
            env.render(ax1)
            ax2.clear()
            ax2.plot(scores, label='Score', color='tab:blue')
            ax2.plot(losses, label='Loss', color='tab:red', alpha=0.7)
            ax2.plot(epsilons, label='Epsilon', color='tab:green', linestyle='--')
            ax2.set_title('Training Progress')
            ax2.set_xlabel('Episode')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

    plt.ioff()
    plt.show()
    print("Training completed. Best score:", best_score)
    return policy_net, target_net, env, scores, losses


# -------------------------------
# 7. 测试与路径回放动画
# -------------------------------
def visualize_policy(model, env, device='cpu'):
    model.eval()
    state = env.reset()
    fig, ax = plt.subplots(figsize=(6, 6))

    def animate(frame):
        if frame == 0:
            env.reset()
        if env.agent_pos != env.end:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).max(1).indices.item()
            env.step(action)
        ax.clear()
        env.render(ax=ax, show_path=True)
        return []

    ani = FuncAnimation(fig, animate, frames=50, interval=300, blit=False, repeat=False)
    plt.show()
    return ani


# -------------------------------
# train
# -------------------------------
if __name__ == "__main__":

    policy_net, target_net, env, scores, losses = train_dqn(
        grid_size=10,
        num_obstacles=5,
        episodes=2000,
        batch_size=64,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5000,
        tau=0.005,
        lr=3e-4,
        device='cpu'
    )

    # policy_net.load_state_dict(torch.load("best_dqn_grid.pth"))
    # print("Playing best policy...")
    # visualize_policy(policy_net, env, device='cpu')