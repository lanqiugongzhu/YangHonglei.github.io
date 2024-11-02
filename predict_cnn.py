import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn  # 导入 nn 模块以定义模型

# 1. 定义模型结构（与训练时一致）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 加载整个模型
loaded_model = torch.load('cnn_model.pth')
loaded_model.eval()  # 切换到评估模式

# 3. 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model.to(device)  # 将模型移动到设备

# 4. 加载和预处理图像
# 让用户输入图像路径
image_path = input("请输入图像路径: ")

image = Image.open(image_path)  # 使用用户输入的图像路径
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 图像大小为 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
image = transform(image).unsqueeze(0)  # 添加 batch 维度
image = image.to(device)  # 将图像也移动到同一设备

# 5. 使用模型进行预测
output = loaded_model(image)
_, predicted = torch.max(output.data, 1)

# 6. 打印预测结果
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
print(f'Predicted class: {classes[predicted.item()]}')
