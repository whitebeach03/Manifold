import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. データの読み込み
data = np.load('all_generated_high_dim_data.npy')  # Shape: (5000, 96, 96)
labels = np.load('all_labels.npy')  # Shape: (5000,)

# 2. データをTorchテンソルに変換
data_tensor = torch.tensor(data, dtype=torch.float32)  # Float型のTensor
labels_tensor = torch.tensor(labels, dtype=torch.long)  # Long型（整数）のTensor

# 3. PyTorchデータセットの作成
dataset = TensorDataset(data_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

for images, labels in enumerate(dataloader):
    print(images)
    print(labels)
    break