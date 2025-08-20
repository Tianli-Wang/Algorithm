import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        # print(state)
        # state = torch.tensor([state], dtype=torch.float).to(self.device)  # 添加批次维度
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)

        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    # def update(self, transition_dict):
        
    #     states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)

    #     actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
    #     rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
    #     next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
    #     dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

    #     td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
    #     td_delta = td_target - self.critic(states)

    #     log_probs = torch.log(self.actor(states)).gather(1, actions) ## actions形状通常为(batch_size, 1)
    #     actor_loss = torch.mean(-log_probs * td_delta)

    #     # critic_loss = torch.mean(F.mse_loss(states), td_delta)

    #     critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

    #     self.actor_optimizer.zero_grad()
    #     self.critic_optimizer.zero_grad()
    #     actor_loss.backward()
    #     critic_loss.backward()
    #     self.actor_optimizer.step()
    #     self.critic_optimizer.step()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # TD 目标和误差
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        # 策略梯度
        log_probs = torch.log(self.actor(states)).gather(1, actions)
        actor_loss = torch.mean(-log_probs * td_delta.detach())  # 注意：td_delta.detach()

        # 价值函数损失
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())

        # 反向传播
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(device)

env_name = 'CartPole-v1'
env = gym.make(env_name)

# state, _ = env.reset()  # 新版返回元组 (state, info)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# print(state)
# print(state_dim)
# print(action_dim)

agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

# 绘制结果
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

# 绘制移动平均曲线
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Moving Average Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()