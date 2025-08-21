# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import gym
# from tensorboardX import SummaryWriter

# env = gym.make('MountainCar-v0')

# num_state = env.observation_space.shape()
# MAX_EPISODE = 400000
# MAX_STEPS = 10000

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # full connected nets
#         self.fc1 = nn.Linear(num_state, 100)
#         self.fc2 = nn.Linear(100, num_state)
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         action_values = self.fc2(x)
#         return action_values
        

# class DQN():
#     capacity = 8000
#     def __init__(self):
#         self.num_actions = env.action_space.n
#         self.eval, self.target = Net(), Net()

#     def choose_action()


# def main():
#     agent = DQN()
#     for i_epi in range (MAX_EPISODE):
#         state = env.reset()
#         for steps in range (MAX_STEPS):
#             action = choose_action(state)  # TODO
#             next_state, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated

#             store_transition(state, action, reward, next_state)  # TODO
#             state = next_state

#             if done or steps >= MAX_STEPS - 1:
#                 agent.update_net()  # TODO
#                 agent.writer.add_scalar()  # TODO
#                 if i_epi % 10 == 0:
#                     print(f"episodes: {i_epi}, steps: {steps}")
#                 break



import torch
import torch.nn.functional as F
import gym
import numpy as np
import collections
import random
import rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 模型保存路径
MODEL_PATH = "dqn_cartpole.pth"
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)
    

class QNet(torch.nn.Module):
    def __init__(self, state_dim, hiddeb_dim, action_dim):
        super(QNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hiddeb_dim)
        self.fc2 = torch.nn.Linear(hiddeb_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, device,target_update_freq):
        self.action_dim = action_dim
        self.q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.device = device
        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1,actions)
        ## max next state q values
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
        q_targets = rewards + self.gamma *max_next_q_values * (1- dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        
        self.optimizer.zero_grad() ## 梯度清零
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_q_net.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")
            return True
        else:
            print(f"No model found at {path}, starting from scratch.")
            return False


lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name = 'CartPole-v1'
env = gym.make(env_name)

replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, device, target_update)

return_list = []

for i in range(10):
    with tqdm(total = int(num_episodes / 10), desc = 'Iteration %d' % i) as pbar:

        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state, info = env.reset()
            done = False

            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)

            return_list.append(episode_return)

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

agent.save(MODEL_PATH)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()