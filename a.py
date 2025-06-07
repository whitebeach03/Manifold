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
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet
from sklearn.manifold import TSNE
from foma import foma, foma_hard
from torch.utils.data import Sampler

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = STL10(root="./data", split="test",  download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)


# 最初のバッチを取得
images, labels = next(iter(loader))
num_classes = 10
alpha = 1.0
rho = 0.9

# FOMA適用
X_scaled, soft_labels = foma(images, labels, num_classes, alpha, rho)

# 逆正規化（CIFAR-10の場合）
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img * std + mean

# 各クラスごとに最初の1枚を選択
class_indices = []
for class_id in range(num_classes):
    indices = (labels == class_id).nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        class_indices.append(indices[0])

plt.figure(figsize=(15, 6))
for idx, i in enumerate(class_indices):
    # 元画像
    plt.subplot(2, num_classes, idx + 1)
    img = denormalize(images[i].cpu())
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f'Original\nLabel: {labels[i].item()}')

    # FOMA画像
    plt.subplot(2, num_classes, num_classes + idx + 1)
    img_foma = denormalize(X_scaled[i].detach().cpu())
    plt.imshow(img_foma.permute(1, 2, 0).numpy())
    plt.axis('off')
    max_prob, pred_class = torch.max(soft_labels[i], dim=0)
    plt.title(f'FOMA\nLabel: {pred_class.item()}\n{max_prob.item():.3f}')

plt.tight_layout()
plt.savefig("./CIFAR10_FOMA_per_class.png")

# ソフトラベル出力（丸め済み）
for idx, i in enumerate(class_indices):
    probs = soft_labels[i].detach().cpu().numpy()
    rounded_probs = [round(p, 3) for p in probs]
    print(f"Class {labels[i].item()} soft label (probabilities): {rounded_probs}")


