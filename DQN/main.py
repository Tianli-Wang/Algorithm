import random
 
import gym
#导入游戏库
import numpy as np
import torch
import torch.nn as nn
 
from agent import Agent
 
 
env = gym.make("CartPole-v1")
#创建一个游戏环境
s=env.reset()#初始化游戏，返回一组观测值，s是一个向量
n_state=len(s)#这里是输入的个数
n_action=env.action_space#这里是输出的动作对应状态
 
EPSILON_DECAY=10000#假设可以衰减10000次
 
EPSILON_START=1.0#探索率从1开始衰减
EPSILON_END=0.02#探索率的最低值
n_episode=5000
#假设玩5000句
n_time_step=1000
#假设每局1000步
agent=Agent(n_input=n_state,n_output=n_action)
#实例化一个对象
TARGET_UPDATE_FREQUENCY=10
#q_target和q网络的同步频率为10，即10局更新一次
REWARD_BUFFER=np.empty(shape=n_episode)
 
for episode_i in range(n_episode):
    for step_i in range(n_time_step):
        episode_reward = 0#奖励值从0开始
        epsilon = np.interp(episode_i * n_time_step+step_i,[0,EPSILON_DECAY],[EPSILON_START,EPSILON_END])#现在开始初始化探索率，探索率随着时间递减，最后保持不变
        random_sample=random.random()#表示从0到1之间随机取一个数
 
        if random_sample<=epsilon:
            a=env.action_space.sample()
        else:
            a=agent.online_net.action(s)#TODO
 
        s_,r,done,info = env.step(a)
        agent.memeo.add_memo(s,a,r,done,s_)#TODO
        s=s_
        episode_reward+=r#每一个回合有一个累计奖励的过程
 
        if done:#判断本局是否结束
            s=env.reset()#如果结束，那就重置环境
            REWARD_BUFFER[episode_i]=episode_reward  #把这一句累加的回合奖励记录下来
            break
        batch_s,batch_a,batch_r,batch_done,bacth_s_=agent.memo.sample()#通过这样一个采样函数，将结果赋值给batch_S
        #计算targets
        target_q_values=agent.target_net(batch_s)#TODO
        max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]#得到最大的target数值
        targets=batch_r+agent.GAMMA *(1-batch_done)*max_target_q_values#这一步就是计算公式#TODO
 
        #计算q_values
        q_values=agent.online_net(batch_s)#给Q网络输入状态S，然后#TODO
        a_q_values=torch.gather(input=q_values,dim=1,index=batch_a)#思考一下为什么是选择第二列，还有序号为batch_a？
        #这一步做的是将所有的q_value给搜集起来，按顺序把各batch与对应各自的最大值q匹配起来放到一起。
 
        #计算loss函数
        loss=nn.functional.smooth_l1_loss(targets,a_q_values)#这下就求出了loss
 
        #计算梯度下降，来更新神经网络中的各个参数
        agent.optimizer.zero_grad()#TODO
        loss.backward()
        agent.optimizer.step()#通过这几步完成一次梯度下降#TODO
 
    if episode_i % TARGET_UPDATE_FREQUENCY==0:
        agent.target_net.load_state_dict (agent.online_net.state_dict())#这一步将online_net网络中的参数全部同步到target_net网络之中
 
        #观测每一个回合的奖励，看看训练的咋样了
        print("Episode:{}".format(episode_i))
        print("Avg.Reward:{}".format(np.mean(REWARD_BUFFER[:episode_i])))
 