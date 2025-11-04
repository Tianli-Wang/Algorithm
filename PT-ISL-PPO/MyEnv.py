# MyEnv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class NonUniformGridWorldEnv(gym.Env):
    """
    一个非均匀网格世界环境。
    - 观测空间 (Dict):
        - "agent_pos": (row, col)
        - "surroundings": [上, 下, 左, 右] 是否被阻挡 (0=通路, 1=阻挡)
    - 动作空间 (Discrete): 4 (上, 下, 左, 右)
    - 奖励: 
        - 成功移动: -time_cost
        - 失败移动: -time_cost + fail_penalty
        - 撞墙/障碍: wall_penalty
        - 奖励塑形: +/- manhattan_reward
        - 到达终点: +goal_reward
    """

    metadata = {'render_modes': ['console']}

    def __init__(self, grid_size=10, render_mode='console', seed=None, obstacle_ratio=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.obstacle_ratio = obstacle_ratio

        self.action_space = spaces.Discrete(4)
        self.action_map = {
            0: (-1, 0),  
            1: (1, 0),   
            2: (0, -1),  
            3: (0, 1)    
        }

        # 观测空间定义为字典，包含agent_pos和surroundings
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            "surroundings": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int32)
        })

        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

        self._initialize_dynamics(seed)

        self.goal_reward = 500.0
        self.wall_penalty = -10.0
        self.fail_penalty = -5.0
        self.manhattan_reward = 1.0    # 显著增大塑形奖励
        self.step_penalty = -0.5       # 每步的小惩罚,鼓励快速到达

        self.agent_pos = None

    def _get_obs(self):
        """获取当前的字典观测值。"""
        surroundings = [0, 0, 0, 0] # 对应 [上, 下, 左, 右]
        current_pos = self.agent_pos

        for action, move in self.action_map.items():
            neighbor_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            
            # 检查是否越界 (墙)
            if not (0 <= neighbor_pos[0] < self.grid_size and 0 <= neighbor_pos[1] < self.grid_size):
                surroundings[action] = 1 # 阻挡
            # 检查是否是障碍物
            elif neighbor_pos in self.obstacles:
                surroundings[action] = 1 # 阻挡

        return {
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "surroundings": np.array(surroundings, dtype=np.int32)
        }


    # def _pos_to_obs(self, pos):
    #     return pos[0] * self.grid_size + pos[1]
    
    # def _obs_to_pos(self, obs):
    #     return (obs // self.grid_size, obs % self.grid_size)
    
    def _initialize_dynamics(self, seed=None):
        """初始化概率、时间和障碍物。"""
        rng = np.random.default_rng(seed)
        self.possibility_matrix = rng.uniform(0.6, 1.0, (self.grid_size, self.grid_size, 4))
        self.time_matrix = rng.integers(1, 5, (self.grid_size, self.grid_size, 4))

        # generate obstacles
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

        self.agent_pos = self.start_pos
        # return self._pos_to_obs(self.agent_pos), {}
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        current_pos = self.agent_pos

        current_dist = abs(current_pos[0] - self.goal_pos[0]) + abs(current_pos[1] - self.goal_pos[1])

        move = self.action_map[action]
        next_pos_candidate = (current_pos[0] + move[0], current_pos[1] + move[1])

        next_pos = current_pos
        reward = self.step_penalty
        moved_successfully = False

        # 检查是否撞墙或障碍物
        if not (0 <= next_pos_candidate[0] < self.grid_size and 0 <= next_pos_candidate[1] < self.grid_size):
            reward += self.wall_penalty
        elif next_pos_candidate in self.obstacles:
            reward += self.wall_penalty
        
        # 尝试移动
        else:   
            prob = self.possibility_matrix[current_pos[0], current_pos[1], action]
            time_cost = self.time_matrix[current_pos[0], current_pos[1], action]

            if self.np_random.random() < prob:
                next_pos = next_pos_candidate
                reward += -time_cost
                moved_successfully = True
            else:
                # 失败移动
                reward += -time_cost + self.fail_penalty
       
        self.agent_pos = next_pos

        # 奖励塑形
        new_dist = abs(next_pos[0] - self.goal_pos[0]) + abs(next_pos[1] - self.goal_pos[1])

        # 只有成功移动时才给予方向奖励
        if moved_successfully:
            if new_dist < current_dist:
                reward += self.manhattan_reward  # 接近目标
            elif new_dist > current_dist:
                reward -= self.manhattan_reward  # 远离目标

        terminated = (self.agent_pos == self.goal_pos)
        if terminated:
            reward += self.goal_reward # 到达终点

        return self._get_obs(), float(reward), terminated, False, {}


    # def step(self, action):
    #     action = int(action)
    #     current_pos = self.agent_pos

    #     # manhattan distance to goal
    #     current_dist = abs(current_pos[0] - self.goal_pos[0]) + abs(current_pos[1] - self.goal_pos[1])

    #     move = self.action_map[action]
    #     next_pos_candidate = (current_pos[0] + move[0], current_pos[1] + move[1])

    #     next_pos = current_pos
    #     reward = 0.0

    #     # check if out of bounds
    #     if not (0 <= next_pos_candidate[0] < self.grid_size and 0 <= next_pos_candidate[1] < self.grid_size):
    #         reward = self.wall_penalty

    #     # check if is obstacle
    #     elif next_pos_candidate in self.obstacles:
    #         reward = self.wall_penalty
        
    #     # try to move
    #     else:   
    #         prob = self.possibility_matrix[current_pos[0], current_pos[1], action]
    #         time_cost = self.time_matrix[current_pos[0], current_pos[1], action]

    #         if self.np_random.random() < prob:
    #             next_pos = next_pos_candidate # only successd to change position
    #             reward = -time_cost  # time cost
    #         else:
    #             # failed to move + time cost
    #             reward = self.fail_penalty - time_cost
        
    #     if next_pos == current_pos:
    #         reward += self.stay_penalty # if stay in place, penalty with time cost

    #     self.agent_pos = next_pos

    #     # reward shaping
    #     new_dist = abs(next_pos[0] - self.goal_pos[0]) + abs(next_pos[1] - self.goal_pos[1])
    #     if new_dist < current_dist:
    #         reward += self.manhattan_reward 
    #     else:
    #         reward -= self.manhattan_reward

    #     terminated = (self.agent_pos == self.goal_pos)
    #     if terminated:
    #         reward += self.goal_reward

    #     return self._pos_to_obs(self.agent_pos), float(reward), terminated, False, {}
    
    def render(self):
        if self.render_mode == 'console':
            print("\033[H\033[J", end="")  # clear screen
            grid = np.full((self.grid_size, self.grid_size), '_', dtype=str)
            for (i, j) in self.obstacles:
                grid[i, j] = 'X'
            grid[self.start_pos] = 'S'
            grid[self.goal_pos] = 'G'
            grid[self.agent_pos] = 'A'
            for row in grid:
                print(' '.join(row))