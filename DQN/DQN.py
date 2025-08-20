import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from tensorboardX import SummaryWriter

env = gym.make('MountainCar-v0')

num_state = env.observation_space.shape()
MAX_EPISODE = 400000
MAX_STEPS = 10000

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # full connected nets
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_state)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_values = self.fc2(x)
        return action_values
        

class DQN():
    capacity = 8000
    def __init__(self):
        self.num_actions = env.action_space.n
        self.eval, self.target = Net(), Net()

    def choose_action()


def main():
    agent = DQN()
    for i_epi in range (MAX_EPISODE):
        state = env.reset()
        for steps in range (MAX_STEPS):
            action = choose_action(state)  # TODO
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            store_transition(state, action, reward, next_state)  # TODO
            state = next_state

            if done or steps >= MAX_STEPS - 1:
                agent.update_net()  # TODO
                agent.writer.add_scalar()  # TODO
                if i_epi % 10 == 0:
                    print(f"episodes: {i_epi}, steps: {steps}")
                break

