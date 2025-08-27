import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import rl_utils
from tqdm import tqdm



import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.std = torch.nn.Linear(hidden_dim, action_dim)  # 输出标准差

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc2(x))
        std = F.softplus(self.std(x)) 
        return mu, std

class ValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, gamma, epochs, eps_clip, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNetContinuous(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.lmbda = lmbda
        self.gamma = gamma
        self.epochs = epochs
        self.eps_clip = eps_clip
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        mu, std = self.actor(state)
        action_dict = torch.distributions.Normal(mu, std)
        action = action_dict.sample()
        # print('here')

        return action.detach().cpu().numpy().flatten()
    
    def update(self, transition_dict):
        # 高效转换：list of arrays → single tensor
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        rewards = (rewards + 8.0) / 8.0  # 奖励缩放
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)

        ## difference from PPO_discrete
        old_mu, old_std = self.actor(states)
        old_action_dicts = torch.distributions.Normal(old_mu, old_std)
        old_log_probs = old_action_dicts.log_prob(actions)

        ## 小步近端更新，利用minibatch数据多次更新梯度
        # for _ in range(self.epochs):
        #     mu, std = self.actor(states)
        #     action_dicts = torch.distributions.Normal(mu, std)
        #     log_probs = action_dicts.log_prob(actions)
        #     ratio = torch.exp(log_probs - old_log_probs)

        #     surr1 = ratio * advantage
        #     surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        #     actor_loss = torch.mean(-torch.min(surr1, surr2))
        #     critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        #     self.actor_optimizer.zero_grad()
        #     self.critic_optimizer.zero_grad()
        #     actor_loss.backward()
        #     critic_loss.backward()
        #     self.actor_optimizer.step()
        #     self.critic_optimizer.step()

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dicts = torch.distributions.Normal(mu, std)
            log_probs = action_dicts.log_prob(actions).sum(dim=-1, keepdim=True)  # ✅ 加 sum
            ratio = torch.exp(log_probs - old_log_probs.detach())  # ✅ 加 detach

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            # 清空梯度
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # 一次 backward（先 actor，再 critic，或合并）
            actor_loss.backward()
            critic_loss.backward()

            # 更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                # state = env.reset()
                state, info = env.reset()  # 只取第一个元素作为状态

                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    with torch.no_grad():
        # td_delta(T, 1)
        T = td_delta.size(0)
        advantage = torch.zeros_like(td_delta)
        adv = 0
        #calculate advantage reverse
        for t in range(T-1, -1, -1):
            adv = td_delta[t] + gamma * lmbda * adv
            advantage[t] = adv
        return advantage

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'Pendulum-v1'
env = gym.make(env_name)
# env.seed(0)
# torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, gamma, epochs, eps, device)

return_list = train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 21)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()