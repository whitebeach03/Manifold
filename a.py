import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.cm as cm
import pickle
import argparse
import matplotlib.pyplot as plt
from src.utils import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet
from sklearn.manifold import TSNE
from foma import foma, foma_hard
from torch.utils.data import Sampler

# --- 変換定義 ---
default_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# オリジナル画像用（ToTensor + Normalizeのみ）
original_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# データセットの読み込み
original_dataset = CIFAR100(root="./data", train=True, transform=original_transform, download=True)
default_dataset = CIFAR100(root="./data", train=True, transform=default_transform, download=True)

# DataLoader（シャッフルなしで同じ順に）
original_loader = DataLoader(original_dataset, batch_size=128, shuffle=False)
default_loader = DataLoader(default_dataset, batch_size=128, shuffle=False)

# 最初のバッチを取得
original_images, labels = next(iter(original_loader))
default_images, _ = next(iter(default_loader))  # ラベルは同じなので省略

# 可視化対象クラス数
num_classes = 10

# CIFAR用の逆正規化関数
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img * std + mean

# 各クラスから最初の1枚ずつ選択
class_indices = []
for class_id in range(num_classes):
    indices = (labels == class_id).nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        class_indices.append(indices[0])

# 可視化：2行（上：Original、下：Default）
plt.figure(figsize=(15, 6))
for idx, i in enumerate(class_indices):
    # オリジナル画像
    plt.subplot(2, num_classes, idx + 1)
    img = denormalize(original_images[i].cpu())
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f'Original\nLabel: {labels[i].item()}')

    # 変換後画像（default_transform）
    plt.subplot(2, num_classes, num_classes + idx + 1)
    img_default = denormalize(default_images[i].cpu())
    plt.imshow(img_default.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f'Default\nLabel: {labels[i].item()}')

plt.tight_layout()
plt.savefig("./CIFAR100_comparison.png")
plt.show()
