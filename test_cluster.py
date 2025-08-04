import os
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet
from torchvision.datasets import CIFAR10, CIFAR100, STL10

# === 設定 ===
data_type  = "cifar100"
model_type = "wide_resnet_28_10"
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

augmentations = [
#     "Default",
#     "Mixup", 
    # "Mixup-FOMA",
    "FOMA-Mixup"
]

# === データ準備 ===
if data_type == "stl10":
    num_classes = 10
    batch_size  = 64
elif data_type == "cifar100":
    epochs      = 400
    num_classes = 100
    batch_size  = 128
elif data_type == "cifar10":
    epochs      = 250
    num_classes = 10
    batch_size  = 128

if data_type == "stl10":
    transform     = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset  = STL10(root="./data", split="train", download=True, transform=transform)
elif data_type == "cifar100":
    transform     = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset  = CIFAR100(root="./data", train=False, transform=transform, download=True)
elif data_type == "cifar10":
    transform     = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset  = CIFAR10(root="./data", train=False, transform=transform, download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# === 各手法に対して評価 ===
for augment in augmentations:
    print(f"\nEvaluating augmentation: {augment}")

    # モデル読み込み
    if model_type == "resnet18":
        model = ResNet18().to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
    
    model_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_0.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 特徴抽出
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False, desc="Extracting features"):
            images = images.to(device)
            features = model.extract_features(images)  # 高次元特徴
            features_list.append(features.cpu())
            labels_list.append(labels)

    X = torch.cat(features_list, dim=0).numpy() 
    y = torch.cat(labels_list, dim=0).numpy()

    silhouette = silhouette_score(X, y)
    ch_score    = calinski_harabasz_score(X, y)

    # 結果出力
    print(f"Silhouette Score         : {silhouette:.4f}")
    print(f"Calinski-Harabasz Index  : {ch_score:.2f}")
