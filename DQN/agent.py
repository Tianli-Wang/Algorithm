import random
import numpy as np
import torch
import torch.nn as nn #DQN中需要用到，torch中神经网络这个模块。
class Replaymemory():
    def __init__(self,n_s,n_a):
        self.n_s=n_s
        self.n_a=n_a
        self.MEMORY_SIZE=1000#经验池的大小
        self.BATCH_SIZE=64#每次批量的大小
 
        self.all_s=np.empty(shape=(self.MEMORY_SIZE,self.n_s),dtype=np.float32)
        #开辟一个空白空间，形状是2维的，与MEMORY_SIZE和n_s共同相关，n_s不一定清楚，且数据类型是32位的。
        self.all_a=np.random.randint(low=0,high=n_a,size=self.MEMORY_SIZE,dtype=np.uint8)
        # 这个游戏中，动作只有左和右两个状态，因此可以使用0和1代替,这里使用n_a变量，在赋值时令为2就可以。
        self.all_r=np.empty(self.MEMORY_SIZE,dtype=np.float32)
        #只是一维的存储空间，因为存储的只是每一局导出的数据。
        self.all_done = np.random.randint(low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_s_=np.empty(shape=(self.MEMORY_SIZE,self.n_s),dtype=np.float32)
        self.t_memo=0
        self.t_max=0#在这里做一个变量的标记
    #这一下子，上边就准备好了存储空间。后续直接进行调用就可以。#
 
    def add_memo(self,s,a,r,done,s_):
        #在这里对t_max进行一个判别
 
        self.all_s[self.t_memo]=s
        self.all_a[self.t_memo]=a
        self.aal_r[self.t_memo]=r
        self.all_done[self.t_memo]=done
        self.all_s_[self.t_memo]=s_
        self.t_max = max(self.t_memo, self.t_memo + 1)
        self.t_memo=(self.t_memo+1)%self.MEMORY_SIZE#检查是否超过1000，如果是第1001，那么取余之后，会放在第一个位置
 
 
 
 
    def sample(self):#经验池中，大于64个经验的话，随机取64个，小于64个的话，有几个取几个。
        if self.t_max>self.BATCH_SIZE:
            idxes = random.sample(range(0,self.t_max), self.BATCH_SIZE)  # 从经验池空间的序号中，随机选择64个序号
        else:
            idxes=range(0,self.t_max)
        batch_s=[]
        batch_a=[]
        batch_r=[]
        batch_done=[]
        batch_s_=[]
        for idx in idxes:#遍历前面的随机取出的64个数据，装载到batch_s这边的5个类型数据中
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])
            #torch识别不了numpy，所以需要把numpy转化成torch。
        batch_s_tensor=torch.as_tensor(np.asarray(batch_s),dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)#最后一句是升维
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)
 
        return batch_s_tensor,batch_a_tensor,batch_r_tensor,batch_done_tensor,batch_s__tensor
 
 
class DQN(nn.Module):
    def __init__(self,n_input,n_output):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(in_features=n_input,out_features=88),
            nn.Tanh(),
            nn.Linear(in_features=88,out_features=n_output)
        )
    def forward(self,x):#前向传播函数
        return self.net(x)
 
    def act(self,obs):#得到输出的动作
        obs_tensor=torch.as_tensor(obs,dtype=torch.float32)#先将数据转化成torch格式
        q_value = self(obs_tensor.unsqueeze(0))#把他转化成行向量
        max_q_idx=torch.argmax(input=q_value)#然后求他最大的q_value对应的序号
        action = max_q_idx.detach().item()#找到这个序号对应的action
        return action#把这个序号对应的操作返回给主程序，执行对应操作。
 
class Agent:
    def __init__(self,n_input,n_output):#在这里定义一下需要的输入和输出数量
        self.n_input=n_input
        self.n_output=n_output
 
        self.GAMA=0.99#这里衰减因子设置为0.99
        self.learning_rate=1e-3#这里设置的学习率
        self.memo=Replaymemory(self.n_input,self.n_output)#TODO,这里是要补充经验池
        self.online_net= DQN(self.n_input,self.n_output)       #TODO 这里是在线训练网络,这两步操作都叫做实例化
        self.target_net= DQN(self.n_input,self.n_output)       #TODO 这里是目标训练网络
 
        self.optimizer= torch.optim.Adam(self.online_net.parameters(),lr=self.learning_rate)#TODO 这里是优化器，优化这个网络参数