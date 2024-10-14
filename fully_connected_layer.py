import torch
import torch.nn as nn

# 创建一个模拟输入张量 (batch_size=2, input_features=3)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# 定义一个全连接层 (input_features=3, output_features=2)
fc_layer = nn.Linear(in_features=3, out_features=2)

# 打印初始化的权重和偏置
print("初始权重:")
print(fc_layer.weight)
print("\n初始偏置:")
print(fc_layer.bias)

# 执行全连接层计算
output = fc_layer(x)
print("\n全连接层输出:")
print(output)
