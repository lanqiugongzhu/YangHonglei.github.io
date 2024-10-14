import torch
from torch.utils.data import Dataset, DataLoader

# 定义一个自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回指定索引的数据和对应的标签
        return self.data[idx], self.labels[idx]

# 模拟一些数据 (5个样本，每个样本有4个特征)
data = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0],
                     [9.0, 10.0, 11.0, 12.0],
                     [13.0, 14.0, 15.0, 16.0],
                     [17.0, 18.0, 19.0, 20.0]])
labels = torch.tensor([0, 1, 0, 1, 0])  # 模拟标签

# 创建自定义数据集和 DataLoader
custom_dataset = MyDataset(data, labels)
custom_loader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

# 打印每个批次的数据
for batch_data, batch_labels in custom_loader:
    print(f"数据: {batch_data}, 标签: {batch_labels}")
