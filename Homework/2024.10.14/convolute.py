import torch
import torch.nn.functional as F

# 创建一个输入张量 (1个批次, 1个通道, 5x5大小)
input_tensor = torch.tensor([[
    [1.0, 2.0, 3.0, 0.0, 1.0],
    [0.0, 1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, 2.0, 1.0],
    [1.0, 3.0, 1.0, 0.0, 2.0],
    [2.0, 0.0, 1.0, 3.0, 4.0]
]]).unsqueeze(0)  # (batch_size=1, channels=1, height=5, width=5)

# 创建一个卷积核 (1个输入通道, 1个输出通道, 3x3大小)
kernel = torch.tensor([[
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0]
]]).unsqueeze(0)  # (out_channels=1, in_channels=1, height=3, width=3)

# 使用 F.conv2d 进行卷积操作
output = F.conv2d(input_tensor, kernel, stride=1, padding=1)  # 保持输出大小一致

print("输入张量:")
print(input_tensor.squeeze())
print("\n卷积核:")
print(kernel.squeeze())
print("\n卷积输出:")
print(output.squeeze())
