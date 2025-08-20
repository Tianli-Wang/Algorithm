

![image-20250725161259165](C:\Users\Tianl\Documents\PhD\Algorithm\DNN\DNN.assets\image-20250725161259165.png)

神经网络的构建涉及到许多参数，可分为如下：

- 参数：通常指网络内部的参数，即此处就指网络中的权重w和偏置b。

- 超参数：一些其他的外部参数，网路的形状（网络层数、每个层数的节点数）、每个激活函数的类型、学习率、轮回次数、每次轮回训练的样本数等。

## 向前传播

 理想的神经网络是如何实现“万能函数模拟器”的功能呢？说白了，就是将一些非线性的激活函数，经过撕拉抓打扯拽（截取、翻转、伸缩、拼接等），合成一个函数，最终这个函数可以拟合输入和输出的关系。前向传播就是指“输入值”进入神经网络，最终输出一个拟合的“输出值”的过程。以上图为例，将单个样本的3个输入特征送入送入神经网络的输入层后，逐层计算到输出层，最终得到神经网络预测的三个特征。以第一层的第一个神经元为例，计算如下：

![image-20250725161606966](C:\Users\Tianl\Documents\PhD\Algorithm\DNN\DNN.assets\image-20250725161606966.png)

上图中的每根线上都有一个权重w，同时神经元节点也有自己的偏执b，对于这个神经元，其输入和输出的关系是：
$$
y=\sigma(\omega_1x_1+\omega_2x_2+\omega_3x_3+b)
$$
其中的σ是一个非线性的激活函数。使用非线性激活函数的原因是，本身的方程w1x1+w2x2+w3x3+b是线性的，如果不套一层非线性函数，无论经过多少层网络，都相当于一层线性网络（线性的叠加还是线性）。最终前向传播就是将输入值通过这样一层层运算计算出最终输出值的过程 。



## 反向传播

神经网络参数的选取直接决定了这个“万能函数模拟器”的拟合效果。那么如何调节这些参数，使得拟合效果好呢？这时候就涉及到反向传播。初始的时候，网络中的参数都是随机的，经过前向传播，会得到一个预测值，这个预测值和实际的输出值会有差距，可以构建一个损失函数来描述这个差距（比如绝对值函数），每次预测都计算当前的损失函数，根据损失函数逐层退回求梯度，目标是使得损失函数的输出最小，此时网络中的参数就是训练好的了。


在这个过程中，需要设定学习率lr，lr越大，则内部参数的优化速度就会变快，但lr过大时，有可能会使得损失函数越过极小值点，在谷底反复横跳，而lr过小，则训练速度可能过慢不易收敛，在网络的训练开始前选择一个合适的学习率还是比较重要的（PyTorch也封装了自动选择学习率的方法）。

最后介绍两个概念，epoch和batch_size：

- epoch就是轮次，1个epoch表示全部样本进行一次前向传播与反向传播。

- batch_size是一次性喂入网络的训练数据的样本数，在训练网络时，通常分批次将样本喂入网络，降低计算开销。通常batch_size是2的整数次幂。
  



## 直观感受神经网络构建的函数 
   这里以StatQuest的案例，以一个只有一层隐藏层的神经网络个为例，看看神经网络是如何拟合出输入与输出的关系的。

### 数据集
​    案例是拟合药物服用剂量（Dose）和药效（Output）的关系。数据集（训练集）一共就三个数据点，当剂量很小或很大时（Dose=0 or Dose = 1），药效很差（Effectiveness=0)，当剂量适中时（Dose = 0.5），药效才好（Effectiveness = 1），总之就这三个点，如下：

![image-20250725162401236](C:\Users\Tianl\Documents\PhD\Algorithm\DNN\DNN.assets\image-20250725162401236.png)

### 神经网络的结构
​    那么我们的神经网络的目标，就是拟合出一个合理的函数，其函数图像尽可能同时靠近（尽可能穿过）这三个数据点。 最终拟合出的函数并不唯一，与神经网络的结构，激活函数的选择等都有关系，这里假定选用ReLu作为激活函数，构建的神经网络如下：

![image-20250725162424148](C:\Users\Tianl\Documents\PhD\Algorithm\DNN\DNN.assets\image-20250725162424148.png)

这是一个非常简单的神经网络，输入层和输出层之间只有一个隐藏层，隐藏层只有两个神经元，激活函数采用ReLu，注意由于输入仅有一个特征（剂量Dose），且输出也只有一个特征（药效Effectiveness） ，因此与之对应，输入层和输出层也只有一个神经元。

上图中神经网络的参数（即权重w和偏置b）都已经给出，这是训练好的神经网络（即参数是训练好的），最终我们可以用一个测试集进行测试，得到当剂量在0-1之间每隔0.1进行采样，对应的药效Effectiveness输出，最终可以绘制出这样的函数图像：

![image-20250725162441506](C:\Users\Tianl\Documents\PhD\Algorithm\DNN\DNN.assets\image-20250725162441506.png)

 可以看到，这个函数确实能比较好的拟合训练集的参数，本质上讲，这个函数就是由两个非线性的激活函数ReLu经过撕拉抓打扯拽（裁剪、伸缩、翻转、拼接）得到的，至于如何撕拉抓打扯拽，则通过神经网络的参数确定，参数的选择则通过对训练集三个点的反向传播一步一步优化。



### PyTorch实现
​    图中的这个神经网络的参数都是训练优化好的，下面我们简便起见，假设最后一个参数b_final没有优化过，初始化为0，我们尝试用Pytorch实现一下对这个参数的优化，将final_bias初始化为0，看看最终这个-16可否被优化出来的。首先引入一些相关的库：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# avoid intel lib error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class BasicDNN(nn.Module):
    def __init__(self):
        super(BasicDNN, self).__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input):

        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_buttom_relu = input * self.w10 + self.b10
        buttom_relu_output = F.relu(input_to_buttom_relu)
        scaled_buttom_relu_output = buttom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_buttom_relu_output + self.final_bias

        final_output = F.relu(input_to_final_relu)

        return final_output
    
if __name__ == "__main__":
    model = BasicDNN()
    inputs = torch.tensor([0., 0.5, 1.])
    labels = torch.tensor([0., 1., 0.])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = []
    for epoch in range(100):
        total_loss = 0
        for iteration in range(len(inputs)):
            input_i = inputs[iteration]
            label_i = labels[iteration]

            output_i = model(input_i)
            loss = F.mse_loss(output_i, label_i)
            loss.backward()
            total_loss += loss.item()

        if total_loss < 0.0001:
            print(f"Early stopping at epoch {epoch} with loss {total_loss}")
            break

        optimizer.step()
        optimizer.zero_grad()
        losses.append(total_loss)
        print(f"Epoch {epoch}, Loss: {total_loss}")

    Fig = plt.figure()
    plt.plot(range(len(losses)), losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.show(block=True)
```

