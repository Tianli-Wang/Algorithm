import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from PPO.PPO_Continuous import ValueNetContinuous
import rl_utils

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    ## 重采样向前传播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = torch.softplus(self.fc_std(x))
        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob
    
class ValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_out = torch.nn.Linear(action_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class SAC_Continuous():
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = ValueNetContinuous(state_dim, hidden_dim, action_dim).to(device) # 第一个Q网络
        self.critic_2 = ValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_target = ValueNetContinuous(state_dim, hidden_dim, action_dim).to(device) # 第一个目标Q网络
        self.critic_2_target = ValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action, _ = self.actor(state)
        return action.item()
    
    def calculate_target(self, rewards, next_states, dones):
        next_actions, log_probs = self.actor(next_states)
        entropy = -log_probs
        q1_value = self.critic_1_target(next_states, next_actions)
        q2_value = self.critic_2_target(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + torch.exp(self.log_alpha) * entropy + 1e-7
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device)
        rewards = (rewards + 8.0) / 8.0

        ## 计算TD目标
        td_target = self.calculate_target(rewards, next_states, dones)

        critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        ## 更新策略网络
        new_actions, log_probs = self.actor(states)
        entropy = -log_probs
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ## 更新温度参数
        alpha_loss = torch.mean(self.log_alpha.exp() * (entropy - self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.critic_1, self.critic_1_target)
        self.soft_update(self.critic_2, self.critic_2_target)

