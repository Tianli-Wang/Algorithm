# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# class BasicDNN(nn.Module):
#     def __init__(self):
#         super(BasicDNN, self).__init__()
#         self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
#         self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
#         self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
#         self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
#         self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
#         self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

#         self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

#     def forward(self, input):

#         input_to_top_relu = input * self.w00 + self.b00
#         top_relu_output = F.relu(input_to_top_relu)
#         scaled_top_relu_output = top_relu_output * self.w01

#         input_to_buttom_relu = input * self.w10 + self.b10
#         buttom_relu_output = F.relu(input_to_buttom_relu)
#         scaled_buttom_relu_output = buttom_relu_output * self.w11

#         input_to_final_relu = scaled_top_relu_output + scaled_buttom_relu_output + self.final_bias

#         # final_output = input_to_final_relu
#         final_output = F.relu(input_to_final_relu)

#         return final_output
    
# if __name__ == "__main__":
#     model = BasicDNN()
#     inputs = torch.tensor([0., 0.5, 1.])
#     labels = torch.tensor([0., 1., 0.])
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#     losses = []
#     for epoch in range(100):
#         total_loss = 0
#         for iteration in range(len(inputs)):
#             input_i = inputs[iteration]
#             label_i = labels[iteration]

#             output_i = model(input_i)
#             loss = F.mse_loss(output_i, label_i)
#             loss.backward()
#             total_loss += loss.item()

#         if total_loss < 0.0001:
#             print(f"Early stopping at epoch {epoch} with loss {total_loss}")
#             break

#         optimizer.step()
#         optimizer.zero_grad()
#         losses.append(total_loss)
#         print(f"Epoch {epoch}, Loss: {total_loss}")

#     Fig = plt.figure()
#     plt.plot(range(len(losses)), losses, label='Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss Value')
#     plt.show(block=True)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# class BasicDNN(nn.Module):
#     def __init__(self, use_final_relu=True):
#         super(BasicDNN, self).__init__()
#         self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
#         self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
#         self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
#         self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
#         self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
#         self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
#         self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
#         self.use_final_relu = use_final_relu
        
#     def forward(self, input):
#         input_to_top_relu = input * self.w00 + self.b00
#         top_relu_output = F.relu(input_to_top_relu)
#         scaled_top_relu_output = top_relu_output * self.w01
        
#         input_to_buttom_relu = input * self.w10 + self.b10
#         buttom_relu_output = F.relu(input_to_buttom_relu)
#         scaled_buttom_relu_output = buttom_relu_output * self.w11
        
#         input_to_final_relu = scaled_top_relu_output + scaled_buttom_relu_output + self.final_bias
        
#         if self.use_final_relu:
#             final_output = F.relu(input_to_final_relu)
#         else:
#             final_output = input_to_final_relu
            
#         return final_output

# # 生成一些非线性数据
# torch.manual_seed(42)
# x = torch.linspace(-1, 1, 100).view(-1, 1)
# y = x**2  # 二次函数（非线性）

# # 训练两个模型：一个有输出ReLU，一个没有
# models = {
#     'with_relu': BasicDNN(use_final_relu=True),
#     'without_relu': BasicDNN(use_final_relu=False)
# }

# optimizers = {
#     name: torch.optim.SGD(model.parameters(), lr=0.01)
#     for name, model in models.items()
# }

# criterion = nn.MSELoss()
# epochs = 1000
# losses = {name: [] for name in models}

# for epoch in range(epochs):
#     for name, model in models.items():
#         optimizer = optimizers[name]
#         optimizer.zero_grad()
#         y_pred = model(x)
#         loss = criterion(y_pred, y)
#         loss.backward()
#         optimizer.step()
#         losses[name].append(loss.item())
        
#         # 打印梯度信息（仅前10个epoch）
#         if epoch < 10:
#             print(f"Epoch {epoch}, {name}, final_bias_grad: {model.final_bias.grad.item():.4f}, loss: {loss.item():.4f}")

# # 可视化损失曲线
# plt.figure(figsize=(10, 5))
# for name, loss_list in losses.items():
#     plt.plot(loss_list, label=name)
# plt.legend()
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# # 可视化预测结果
# plt.figure(figsize=(12, 5))
# for i, (name, model) in enumerate(models.items(), 1):
#     plt.subplot(1, 2, i)
#     y_pred = model(x).detach()
#     plt.scatter(x.numpy(), y.numpy(), label='True')
#     plt.scatter(x.numpy(), y_pred.numpy(), label='Predicted')
#     plt.title(f'{name}')
#     plt.legend()
# plt.tight_layout()
# plt.show()




import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置随机种子，确保结果可复现
torch.manual_seed(42)

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载训练集和测试集
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 定义基础DNN模型
class BasicDNN(nn.Module):
    def __init__(self):
        super(BasicDNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(128, 64)       # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(64, 10)        # 第二个隐藏层到输出层
        self.relu = nn.ReLU()               # ReLU激活函数
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将28x28的图像展平为一维向量
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)          # 输出层不使用激活函数，因为后面会使用CrossEntropyLoss
        return x

# 初始化模型、损失函数和优化器
model = BasicDNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 计算平均损失
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return train_losses

# 测试函数
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的类别
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return accuracy

# 训练模型
epochs = 5
train_losses = train(model, train_loader, criterion, optimizer, epochs)

# 评估模型
accuracy = test(model, test_loader)

# 可视化训练损失
plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'b-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'basic_dnn_model.pth')
print("Model saved as 'basic_dnn_model.pth'")    




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# # 设置环境变量解决OpenMP冲突问题
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# class BasicDNN(nn.Module):
#     def __init__(self):
#         super(BasicDNN, self).__init__()
#         self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
#         self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
#         self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
#         self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
#         self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
#         self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        
#         # 唯一可训练参数
#         self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

#     def forward(self, input):
#         # 顶部神经元计算
#         input_to_top_relu = input * self.w00 + self.b00
#         top_relu_output = F.relu(input_to_top_relu)
#         scaled_top_relu_output = top_relu_output * self.w01
        
#         # 底部神经元计算
#         input_to_bottom_relu = input * self.w10 + self.b10
#         bottom_relu_output = F.relu(input_to_bottom_relu)
#         scaled_bottom_relu_output = bottom_relu_output * self.w11
        
#         # 最终输出 (移除ReLU以允许负值)
#         input_to_final = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
#         final_output = input_to_final  # 移除ReLU激活函数
        
#         return final_output

# if __name__ == "__main__":
#     # 设置随机种子以确保结果可复现
#     torch.manual_seed(42)
    
#     # 创建模型
#     model = BasicDNN()
    
#     # 准备数据
#     inputs = torch.tensor([0., 0.5, 1.], requires_grad=False).view(-1, 1)
#     labels = torch.tensor([0., 1., 0.], requires_grad=False).view(-1, 1)
    
#     # 优化器
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, 'min', patience=5, factor=0.5
#     )
    
#     # 训练参数
#     max_epochs = 1000
#     losses = []
#     best_loss = float('inf')
#     patience = 20
#     counter = 0
    
#     for epoch in range(max_epochs):
#         # 前向传播
#         outputs = model(inputs)
#         loss = F.mse_loss(outputs, labels)
        
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # 记录损失
#         loss_value = loss.item()
#         losses.append(loss_value)
        
#         # 学习率调度
#         scheduler.step(loss_value)
        
#         # 早停检查
#         if loss_value < best_loss:
#             best_loss = loss_value
#             counter = 0
#         else:
#             counter += 1
            
#         if counter >= patience:
#             print(f"Early stopping at epoch {epoch} with loss {loss_value:.7f}")
#             break
            
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Loss: {loss_value:.7f}")
    
#     # 打印最终结果
#     print(f"Final loss: {best_loss:.7f}")
#     print(f"Final bias: {model.final_bias.item():.7f}")
    
#     # 可视化损失曲线
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(losses)), losses, label='Training Loss')
#     plt.axhline(y=0.0001, color='r', linestyle='--', label='Early Stopping Threshold')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss (MSE)')
#     plt.title('Training Loss over Epochs')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
    
#     # 显示预测结果
#     with torch.no_grad():
#         predictions = model(inputs)
    
#     print("\nPredictions vs Labels:")
#     for i in range(len(inputs)):
#         print(f"Input: {inputs[i].item():.1f}, Predicted: {predictions[i].item():.7f}, Target: {labels[i].item():.1f}")
    
#     # 显示图形
#     plt.show()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# class BasicDNN(nn.Module):
#     def __init__(self):
#         super(BasicDNN, self).__init__()
#         # Parameters for the top ReLU branch
#         self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
#         self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
#         self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

#         # Parameters for the bottom ReLU branch
#         self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
#         self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
#         self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

#         # Final bias, which is the only trainable parameter
#         self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

#     def forward(self, input):
#         # Top ReLU branch calculations
#         input_to_top_relu = input * self.w00 + self.b00
#         top_relu_output = F.relu(input_to_top_relu)
#         scaled_top_relu_output = top_relu_output * self.w01

#         # Bottom ReLU branch calculations
#         input_to_buttom_relu = input * self.w10 + self.b10
#         buttom_relu_output = F.relu(input_to_buttom_relu)
#         scaled_buttom_relu_output = buttom_relu_output * self.w11

#         # Combination and final ReLU
#         input_to_final_relu = scaled_top_relu_output + scaled_buttom_relu_output + self.final_bias
#         final_output = F.relu(input_to_final_relu)

#         return final_output

# if __name__ == "__main__":
#     model = BasicDNN()
#     inputs = torch.tensor([0., 0.5, 1.], dtype=torch.float32) # Specify dtype for consistency
#     labels = torch.tensor([0., 1., 0.], dtype=torch.float32) # Specify dtype for consistency
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#     losses = []
#     num_epochs = 100 # Define number of epochs for clarity

#     print("Starting training...")
#     for epoch in range(num_epochs):
#         # Reset gradients at the beginning of each epoch
#         optimizer.zero_grad()
        
#         # Calculate total loss for the current epoch
#         # You can compute predictions and loss for all inputs at once for efficiency
#         # However, to preserve your original logic of iterating through inputs, we keep it that way.
#         epoch_loss = 0.0
#         for i in range(len(inputs)):
#             input_i = inputs[i].unsqueeze(0) # Add a batch dimension to input
#             label_i = labels[i].unsqueeze(0) # Add a batch dimension to label

#             output_i = model(input_i)
#             loss_i = F.mse_loss(output_i, label_i)
#             loss_i.backward() # Accumulate gradients for each item
#             epoch_loss += loss_i.item()

#         # Perform a single optimization step after all gradients for the epoch are accumulated
#         optimizer.step()

#         losses.append(epoch_loss)

#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

#         if epoch_loss < 0.0001:
#             print(f"Early stopping at epoch {epoch+1} with loss {epoch_loss:.6f}")
#             break

#     print("\nTraining complete!")
#     print(f"Final Loss: {losses[-1]:.6f}")

#     # Plotting the loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(losses)), losses, label='Epoch Loss', color='blue')
#     plt.xlabel('Epoch')
#     plt.ylabel('Total Loss Value')
#     plt.title('Training Loss Over Epochs')
#     plt.grid(True)
#     plt.legend()
#     plt.show()