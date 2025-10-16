import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class NonUniformGridWorldEnv(gym.Env):
    
    def __init__(self, grid_size=5):
        super(NonUniformGridWorldEnv, self).__init__()
        
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right

        self.observation_space = spaces.Discrete(grid_size * grid_size)

        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

        # 所有概率和时间固定
        self.possibility = 0.8
        self.time = 1.0

        # Reward structure
        self.goal_reward = 10
        self.step_penalty = -1
        self.wall_penalty = -5

        self.agent_pos = None

    # transfer 2D-position to 1D-observation
    def _pos_to_obs(self, pos):
        return pos[0] * self.grid_size + pos[1]
    
    # transfer 1D-observation to 2D-position
    def _obs_to_pos(self, obs):
        return (obs // self.grid_size, obs % self.grid_size)
    
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        observation = self._pos_to_obs(self.agent_pos)
        info = {} # info字典可以用来返回调试信息
        return observation, info

    def step(self, action):
        """执行一个动作"""
        current_pos = self.agent_pos
        
        # 将动作映射到方向变化
        action_map = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        move = action_map[action]
        
        # 计算潜在的下一个位置
        next_pos_candidate = (current_pos[0] + move[0], current_pos[1] + move[1])

        # 检查是否撞墙
        if not (0 <= next_pos_candidate[0] < self.grid_size and \
                0 <= next_pos_candidate[1] < self.grid_size):
            # 撞墙，位置不变，给予惩罚
            reward = self.wall_penalty
            next_pos = current_pos
        else:
            # 随机性：根据概率决定移动是否成功
            if random.random() < self.possibility:
                # 移动成功
                next_pos = next_pos_candidate
                reward = -self.time  # 奖励是消耗时间的负数
            else:
                # 移动失败，停在原地
                next_pos = current_pos
                reward = self.fail_penalty

        self.agent_pos = next_pos

        # 检查是否到达终点
        terminated = (self.agent_pos == self.goal_pos)
        if terminated:
            reward += self.goal_reward # 到达终点获得巨大奖励

        truncated = False # 在这个简单环境中我们不设置步数截断
        observation = self._pos_to_obs(self.agent_pos)
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        """在控制台打印网格"""
        if self.render_mode == 'console':
            grid = np.full((self.grid_size, self.grid_size), '_', dtype=str)
            start_obs = self._pos_to_obs(self.start_pos)
            goal_obs = self._pos_to_obs(self.goal_pos)
            
            grid[self._obs_to_pos(start_obs)] = 'S'
            grid[self._obs_to_pos(goal_obs)] = 'G'
            grid[self.agent_pos] = 'A' # Agent
            
            for row in grid:
                print(' '.join(row))
            print("-" * 20)