import torch
import torch.nn.functional as F

# 创建一个张量（模拟网络的输出）
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU 激活函数
relu_output = F.relu(x)
print("ReLU 输出:")
print(relu_output)

# Sigmoid 激活函数
sigmoid_output = torch.sigmoid(x)
print("\nSigmoid 输出:")
print(sigmoid_output)

# Tanh 激活函数
tanh_output = torch.tanh(x)
print("\nTanh 输出:")
print(tanh_output)

# Leaky ReLU 激活函数（负斜率为0.1）
leaky_relu_output = F.leaky_relu(x, negative_slope=0.1)
print("\nLeaky ReLU 输出:")
print(leaky_relu_output)
