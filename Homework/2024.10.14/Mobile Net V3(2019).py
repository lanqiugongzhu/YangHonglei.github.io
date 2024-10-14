import torch
import torch.nn as nn

class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.relu6(x + 3) / 6

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_dim = in_channels // reduction
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(),
            SqueezeExcitation(16),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# 示例用法
model = MobileNetV3Small(num_classes=10)
print(model)
