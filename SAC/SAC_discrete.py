import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
import matplotlib.pyplot as plt
import datetime
from torch.utils.tensorboard import SummaryWriter
import os


# tensorboard --logdir=logs
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/sac_discrete_{current_time}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CriticNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class SAC_Discrete():
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = CriticNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = CriticNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_target = CriticNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2_target = CriticNet(state_dim, hidden_dim, action_dim).to(device)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # soft update do not need optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # do not need Neural Network, it's a learnable super parameter
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.update_step = 0

    def take_action(self, state):
        # .unsqueeze(0) [3]→ [1, 3]
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def calculate_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-7)
        entropy = - torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1_target(next_states)
        q2_value = self.critic_2_target(next_states)
        min_q_value = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_q_value + torch.exp(self.log_alpha) * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)

        return td_target
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        # update critic nets
        td_target = self.calculate_target(rewards, next_states, dones)
        critic_1_q_value = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_value, td_target.detach())) # detach() remain compute graph
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_q_value = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_value, td_target.detach()))
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # update actor net, use policy gradient
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-7)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_q_value = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-torch.exp(self.log_alpha) * entropy - min_q_value)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update alpha
        alpha_loss = torch.mean(self.log_alpha.exp() * (entropy - self.target_entropy).detach()) # Treat as a fixed scalar; its value is used in computation, but does not participate in gradient propagation; the update of α will affect the gradient of the actor, and the update of the actor will also affect the learning of α.
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # update target networks
        self.soft_update(self.critic_1, self.critic_1_target)
        self.soft_update(self.critic_2, self.critic_2_target)

        self.update_step += 1

        if self.update_step % 10 == 0:  

            alpha = torch.exp(self.log_alpha).item()

            with torch.no_grad():
                avg_q1 = q1_value.mean().item()
                avg_q2 = q2_value.mean().item()
                avg_q = (avg_q1 + avg_q2) / 2

            avg_entropy = entropy.mean().item()

            critic_loss = (critic_1_loss.item() + critic_2_loss.item()) / 2

            # write into TensorBoard
            writer.add_scalar("Alpha", alpha, self.update_step)
            writer.add_scalar("Average_Q_Value", avg_q, self.update_step)
            writer.add_scalar("Entropy", avg_entropy, self.update_step)
            writer.add_scalar("Actor_Loss", actor_loss.item(), self.update_step)
            writer.add_scalar("Critic_Loss", critic_loss, self.update_step)
            writer.add_scalar("Alpha_Loss", alpha_loss.item(), self.update_step)


actor_lr = 1e-3
critic_lr = 1e-3
alpha_lr = 1e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
tau = 0.005  # soft update parameter
buffer_size = 10000
minimal_size = 600
batch_size = 81
target_entropy = -1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name)
# random.seed(0)
# np.random.seed(0)
# env.seed(0)
# torch.manual_seed(0)

replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# target_entropy = -np.log(action_dim) * 0.9
print(f"Action Dim: {action_dim}, Target Entropy: {target_entropy:.3f}")

agent = SAC_Discrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)

return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, writer)

writer.close()

episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('SAC on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episode_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format(env_name))
plt.show()