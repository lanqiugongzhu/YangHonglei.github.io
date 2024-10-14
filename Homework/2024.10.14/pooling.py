import torch
import torch.nn.functional as F

# 创建一个模拟输入张量 (1个批次, 1个通道, 4x4 大小)
x = torch.tensor([[
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
]]).unsqueeze(0)  # (batch_size=1, channels=1, height=4, width=4)

print("输入张量:")
print(x.squeeze())

# 最大池化 (2x2 窗口, 步幅为 2)
max_pool_output = F.max_pool2d(x, kernel_size=2, stride=2)
print("\n最大池化输出:")
print(max_pool_output.squeeze())

# 平均池化 (2x2 窗口, 步幅为 2)
avg_pool_output = F.avg_pool2d(x, kernel_size=2, stride=2)
print("\n平均池化输出:")
print(avg_pool_output.squeeze())
