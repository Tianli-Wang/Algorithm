Q-Learning伪代码：

![img](https://pica.zhimg.com/v2-18ca00ecac9fcb0c87efca306b5b2066_1440w.jpg)
$$
Q(s_t,a_t) \gets Q(s_t,a_t)+\alpha [ r_t+\gamma \max Q(s_{t+1},a_t)-Q(s_t,a_t)]
$$

$\gamma$是作用在下一状态所有可能动作对应的$Q$值的最大值前面，也就是对后续状态的预期收益进行折扣，体现了对未来奖励的重视程度，$\gamma$越大，表明越看重未来获得的奖励。




DQN算法流程图：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b94da5b49d3532581abd94cbc14deb99.png#pic_center)

首先，我们DNN的输出值，自然是在给定状态的情况下，执行各action后能得到的Q值。然而事实上我们在很多情况下并不知道最优的Q值是什么，比如自动驾驶、围棋等情况，所以似乎我们没法给出标签。但是什么是不变的呢？**Reward**！
对状态s，执行动作a，那么得到的reward是一定的，而且是不变的！因此需要考虑从reward下手，让预测Q值和真实Q值的比较问题转换成让模型实质上在拟合reward的问题。

前面被target标出来的地方是这一步得到的reward+下一状态所能得到的最大Q值，它们减去这一步的Q值，那么实际上它们的差就是实际reward减去现有模型认为在s下采取a时能得到的reward值。

![image-20250724203826406](C:\Users\Tianl\Documents\PhD\Algorithm\DQN\DQN.assets\image-20250724203826406.png)

现在的问题就已经转换为需要一组训练集，它能够提供一批四元组（s, a, r, s’），其中s’为s执行a后的下一个状态。如果能有这样一个四元组，就能够用来训练DNN了，这就是我们要介绍的**experience reply**。
**Experience Reply**
前面提到我们需要一批四元组（s, a, r, s’）来进行训练，因此我们需要缓存一批这样的四元组到经验池中以供训练之用。由于每次执行一个动作后都能转移到下一个状态，并获得一个reward，因此我们每执行一次动作后都可以获得一个这样的四元组，也可以将这个四元组直接放入经验池中。我们知道这种四元组之间是存在关联性的，因为状态的转移是连续的，如果直接按顺序取一批四元组作为训练集，那么是容易过拟合的，因为训练样本间不是独立的！为解决这个问题，我们可以简单地从经验池中随机抽取少量四元组作为一个batch，这样既保证了训练样本是**独立同分布**的，也使得每个batch**样本量不大**，能加快训练速度。

![img](https://pic1.zhimg.com/v2-3b1bacc4074dbfd34d40b955d9696d6e_1440w.jpg)

<img src="https://pic4.zhimg.com/v2-e05f4913c4c867b28b279455ebddb2c1_1440w.jpg" alt="img" style="zoom:67%;" />



## 交叉熵损失函数

### 离散情况（分类任务）

假设类别数为K，样本i的真实标签为$y_i=(y_{i1},y_{i2},...,y_{iK})$,其中 $y_{ij} $表示样本 i 属于类别 j 的真实情况（通常是独热编码，即只有一个元素为 1，其余为 0 )。模型的预测概率分布为$\hat{y_i}=(\hat{y_{i1}},\hat{y_{i2}},...,\hat{y_{iK}})$, $\hat{y_{ij}}$表示为样本i被预测为类别j的概率，并且$\Sigma_{j=1}^K \hat{y_{ij}}=1$ .

the cross-entropy loss for a single sample is :
$$
L_i=-\Sigma_{j=1}^K y_{ij} \log(\hat{y_ij})
$$
the cross-entropy loss of the entire dataset is:
$$
L=-\frac{1}{N}\Sigma_{i=1}^N \Sigma_{j=1}^K y_{ij}\log(\hat{y_{ij}})
$$

### 连续情况

在一些回归问题的扩展中，也会用到交叉熵的概念，不过形式有所不同。例如在二分类逻辑回归中，当$y\in\{0,1\}$，预测概率为$\hat{y}$时，单个样本的交叉熵损失为：
$$
L_i=-y\log(\hat{y})-(1-y)\log(1-\hat{y})
$$
交叉熵是信息论中的一个重要概念。在信息论里，熵（Entropy）衡量一个随机变量的不确定性。对于一个离散随机变量 X ，其概率分布为 $P(x)$，熵的定义为 $H(P)=-\sum_{x}P(x)\log(P(x))$。
交叉熵 $H(P,Q)=-\sum_{x}P(x)\log(Q(x))$用于衡

量两个概率分布 P 和 Q 之间的差异。在机器学习分类任务中，真实标签的分布就是 P ，模型的预测分布是 Q 。交叉熵的值越小，说明两个分布越接近，也就意味着模型的预测结果越准确。