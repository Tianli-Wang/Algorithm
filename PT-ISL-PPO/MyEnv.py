# MyEnv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class NonUniformGridWorldEnv(gym.Env):
    metadata = {'render_modes': ['console']}

    def __init__(self, grid_size=10, render_mode='console', seed=None, obstacle_ratio=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.obstacle_ratio = obstacle_ratio

        self.action_space = spaces.Discrete(4)
        self.action_map = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        self.observation_space = spaces.Discrete(grid_size * grid_size)

        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

        self._initialize_dynamics(seed)

        self.goal_reward = 10.0
        self.wall_penalty = -5.0
        self.fail_penalty = -2.0
        self.stay_penalty = -3.0

        self.agent_pos = None

    def _pos_to_obs(self, pos):
        return pos[0] * self.grid_size + pos[1]
    
    def _obs_to_pos(self, obs):
        return (obs // self.grid_size, obs % self.grid_size)
    
    def _initialize_dynamics(self, seed=None):
        rng = np.random.default_rng(seed)
        self.possibility_matrix = rng.uniform(0.6, 1.0, (self.grid_size, self.grid_size, 4))
        self.time_matrix = rng.integers(1, 5, (self.grid_size, self.grid_size, 4))

        # 生成障碍物（排除起点和终点）
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        all_positions = [p for p in all_positions if p not in [self.start_pos, self.goal_pos]]
        num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_ratio)
        num_obstacles = min(num_obstacles, len(all_positions))
        
        self.obstacles = set()
        if num_obstacles > 0:
            indices = rng.choice(len(all_positions), size=num_obstacles, replace=False)
            self.obstacles = {all_positions[i] for i in indices}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.agent_pos = self.start_pos
        return self._pos_to_obs(self.agent_pos), {}

    def step(self, action):
        action = int(action)
        current_pos = self.agent_pos
        move = self.action_map[action]
        next_pos_candidate = (current_pos[0] + move[0], current_pos[1] + move[1])

        # 默认：未移动
        next_pos = current_pos
        reward = 0.0

        # 检查是否越界
        if not (0 <= next_pos_candidate[0] < self.grid_size and 0 <= next_pos_candidate[1] < self.grid_size):
            reward = self.wall_penalty
        # 检查是否是障碍物
        elif next_pos_candidate in self.obstacles:
            reward = self.wall_penalty
        else:
            # 尝试移动
            prob = self.possibility_matrix[current_pos[0], current_pos[1], action]
            time_cost = self.time_matrix[current_pos[0], current_pos[1], action]

            if random.random() < prob:
                next_pos = next_pos_candidate
                reward = -time_cost  # 成功移动，只扣时间成本
            else:
                reward = self.fail_penalty  # 移动失败

        if next_pos == current_pos:
            reward += self.stay_penalty

        self.agent_pos = next_pos
        terminated = (self.agent_pos == self.goal_pos)
        if terminated:
            reward += self.goal_reward

        return self._pos_to_obs(self.agent_pos), float(reward), terminated, False, {}

    def render(self):
        if self.render_mode == 'console':
            print("\033[H\033[J", end="")  # 清屏
            grid = np.full((self.grid_size, self.grid_size), '_', dtype=str)
            for (i, j) in self.obstacles:
                grid[i, j] = 'X'
            grid[self.start_pos] = 'S'
            grid[self.goal_pos] = 'G'
            grid[self.agent_pos] = 'A'
            for row in grid:
                print(' '.join(row))
            print("\n按 Ctrl+C 停止 | X = 障碍物")