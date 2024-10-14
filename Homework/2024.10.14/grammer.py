import torch
import torch.nn as nn

# 定义一个全连接层：输入特征数=4，输出特征数=3
fc_layer = nn.Linear(in_features=4, out_features=3)

# 打印权重和偏置
print("权重矩阵:")
print(fc_layer.weight)
print("偏置向量:")
print(fc_layer.bias)

# 创建输入张量并通过全连接层
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # 输入形状 (1, 4)
output = fc_layer(x)
print("输出:")
print(output)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(4, 3)  # 定义一个全连接层

    def forward(self, x):
        return self.fc1(x)  # 前向传播时使用定义的层

# 实例化模型
model = SimpleModel()
print(model)

class Example:
    def __init__(self, value):
        self.value = value  # 使用 self 存储属性

    def print_value(self):
        print(f"Value: {self.value}")

# 实例化对象并调用方法
obj = Example(10)
obj.print_value()

class CallableClass:
    def __call__(self, x):
        return x + 1

obj = CallableClass()
print(obj(5))  # 输出: 6
print(callable(obj))  # 输出: True
