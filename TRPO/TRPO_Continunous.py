import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
from rl_utils import compute_advantage
import rl_utils
import copy

# 连续动作空间，策略网络输出动作的期望和方差
class PolicyNetContinunous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinunous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim) ## 做什么
        self.std = torch.nn.Linear(hidden_dim, action_dim) ## 探索多大

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = 2.0 * torch.tanh(self.fc2(x))  # action range [-2, 2]
        std = F.softplus(self.std(x))  # 保证标准差为正数
        return mean, std
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class TRPO_Continuous:
    def __init__(self, state_dim, hidden_dim, action_dim, lmbda, kl_constraint, alpha, critic_lr, gamma, device):
        self.actor_net = PolicyNetContinunous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_net = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        print("TRPO_Continuous agent init done")

    def take_action(self, states):
        state = torch.tensor(states, dtype=torch.float).unsqueeze(0).to(self.device)
        mean, std = self.actor_net(state)
        action_dict = torch.distributions.Normal(mean, std) ## 创建高斯分布
        action = action_dict.sample()
        print( "action")

        return [action.item()]

    def hessian_matrix_vector_product(self, states, old_action_dists, vector, damping=0.1):
        mean, std = self.actor_net(states)
        new_action_dists = torch.distributions.Normal(mean, std)
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor_net.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor_net.parameters())
        grad2_vector = torch.cat([g.contiguous().view(-1) for g in grad2])
        print("Hessian-vector product")

        return grad2_vector + damping *  vector ## 防止 Hessian 矩阵奇异或病态

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)

        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)

            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            if torch.sqrt(torch.dot(r, r)) < 1e-10:
                break
            beta = torch.dot(r, r) / rdotr
            p = r + beta * p
            rdotr = torch.dot(r, r)

        print("Conjugate gradient result")
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        log_probs = torch.log(actor(states)).gather(1, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        print("Surrogate objective")

        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):
        old_para = torch.nn.utils.parameters_to_vector(self.actor_net.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor_net)

        for i in range(15):
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor_net)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Normal(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)

            if new_obj > old_obj and kl_div < self.kl_constraint:
                print("Line search successed")
                return new_para

        print("Line search failed")

        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor_net)
        grads = torch.autograd.grad(surrogate_obj, self.actor_net.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads])
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists, max_coef * descent_direction)

        print("New parameters")

        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor_net.parameters())

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic_net(next_states) * (1 - dones)

        td_delta = td_target - self.critic_net(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)
        old_log_probs = torch.log(self.actor_net(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor_net(states).detach())
        critic_loss = torch.mean(F.mse_loss(self.critic_net(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数
        # 更新策略函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)
        print("Policy learned")

num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
critic_lr = 1e-2
kl_constraint = 0.00005
alpha = 0.5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'Pendulum-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
# env.seed(0)
# torch.manual_seed(0)
agent = TRPO_Continuous(state_dim, hidden_dim, action_dim, lmbda, kl_constraint, alpha, critic_lr, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()