# import os
# import torch
# import torchvision
# import random
# import torchvision.transforms as transforms
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import argparse
# from matplotlib.colors import ListedColormap
# from torch.utils.data import DataLoader, Subset
# from src.models.resnet import ResNet18
# from sklearn.manifold import TSNE
# from tqdm import tqdm
# from src.models.wide_resnet import Wide_ResNet
# from torchvision.datasets import STL10, CIFAR10, CIFAR100
# from umap import UMAP

# parser = argparse.ArgumentParser()
# parser.add_argument("--i",          type=int, default=0)
# parser.add_argument("--epochs",     type=int, default=250)
# parser.add_argument("--data_type",  type=str, default="cifar100",  choices=["stl10", "cifar100", "cifar10"])
# parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
# parser.add_argument("--k_foma",     type=int, default=0)
# parser.add_argument("--method",     type=str, default="tsne")
# args = parser.parse_args() 

# i          = args.i
# epochs     = args.epochs
# data_type  = args.data_type
# model_type = args.model_type
# k_foma     = args.k_foma
# method     = args.method
# device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.makedirs(f"./result_features/{method}/{data_type}", exist_ok=True)

# augmentations = [
#     # "Default",
#     # "Mixup", 
#     # "Local-FOMA", 
#     # "ES-Mixup",
#     "Mixup-FOMA",
#     # "Mixup-FOMA2",
# ]

# if data_type == "stl10":
#     num_classes = 10
#     batch_size  = 64
# elif data_type == "cifar100":
#     epochs      = 250
#     num_classes = 100  # ← 学習済みモデルは100クラス想定のまま
#     batch_size  = 128
# elif data_type == "cifar10":
#     epochs      = 250
#     num_classes = 10
#     batch_size  = 128

# REPRESENTATIVE_20_FINE = [0, 1, 3, 8, 9, 15, 22, 23, 33, 37,
#                           45, 46, 47, 49, 58, 59, 61, 71, 84, 95]
# REPRESENTATIVE_20_NAMES = [
#     "apple","aquarium_fish","bear","bicycle","bottle","camel","clock","cloud","forest","house",
#     "lawn_mower","leopard","man","orange","pickup_truck","plain","rocket","sea","table","whale"
# ]
# FINE_TO_20IDX = {fine:i for i, fine in enumerate(REPRESENTATIVE_20_FINE)}

# for augment in augmentations:
#     features_list = []
#     labels_list = []
#     if k_foma == 0:
#         model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"
#     else:    
#         model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}_{k_foma}.pth"
    
#     if model_type == "resnet18":
#         model = ResNet18().to(device)
#     elif model_type == "wide_resnet_28_10":
#         model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
    
#     if data_type == "stl10":
#         transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#         test_dataset  = STL10(root="./data", split="train", download=True, transform=transform)
#     elif data_type == "cifar100":
#         transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#         full_dataset  = CIFAR100(root="./data", train=False, transform=transform, download=True)

#         indices = [idx for idx, fine in enumerate(full_dataset.targets) if fine in REPRESENTATIVE_20_FINE]
#         test_dataset = Subset(full_dataset, indices)

#     elif data_type == "cifar10":
#         transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#         test_dataset  = CIFAR10(root="./data", train=False, transform=transform, download=True)

#     test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

#     checkpoint = torch.load(model_save_path, weights_only=True)
#     if 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         model.load_state_dict(checkpoint)

#     model.eval()
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, leave=False):
#             images = images.to(device)
#             features = model.extract_features(images)
#             features_list.append(features.cpu())
#             labels_list.append(labels)
#     X = torch.cat(features_list, dim=0).numpy()
#     y = torch.cat(labels_list, dim=0).numpy()

#     if data_type == "cifar100":
#         y = np.vectorize(FINE_TO_20IDX.get)(y)

#     if method == "tsne":
#         reducer = TSNE(n_components=2, random_state=42, perplexity=30)
#     elif method == "umap":
#         reducer = UMAP(
#             n_components=2,
#             n_neighbors=15,
#             min_dist=0.1,
#             metric='euclidean',
#             random_state=42
#         )
#     X_2d = reducer.fit_transform(X)

#     plt.figure(figsize=(8, 6))
#     if augment == "Default":
#         augment = "Baseline"
#     if data_type == "cifar100":
#         cmap = plt.get_cmap('tab20', 20)
#         scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap, s=10, alpha=0.7)
#         cbar = plt.colorbar(scatter, ticks=np.arange(20))
#         cbar.set_label("Representative class (20)")
#         try:
#             cbar.ax.set_yticklabels(REPRESENTATIVE_20_NAMES)
#         except Exception:
#             pass
#         title = f"{augment}"
#         save_suffix = "20classes"
#     else:
#         scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=6, alpha=1)
#         plt.colorbar(scatter, label="Class label")
#         title = f"t-SNE of ResNet Feature Representations ({augment})"
#         save_suffix = ""

#     plt.title(title)
#     plt.xlabel("Dim 1")
#     plt.ylabel("Dim 2")
#     plt.tight_layout()
#     plt.savefig(f"./result_features/{method}/{data_type}/_{model_type}_{augment}_{k_foma}_{save_suffix}.png")


import os
import torch
import torchvision
import random
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset
from src.models.resnet import ResNet18
from sklearn.manifold import TSNE
from tqdm import tqdm
from src.models.wide_resnet import Wide_ResNet
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from umap import UMAP

parser = argparse.ArgumentParser()
parser.add_argument("--i",          type=int, default=0)
parser.add_argument("--epochs",     type=int, default=250)
parser.add_argument("--data_type",  type=str, default="cifar100",  choices=["stl10", "cifar100", "cifar10"])
parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
parser.add_argument("--k_foma",     type=int, default=0)
parser.add_argument("--method",     type=str, default="tsne")
args = parser.parse_args() 

i          = args.i
epochs     = args.epochs
data_type  = args.data_type
model_type = args.model_type
k_foma     = args.k_foma
method     = args.method
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(f"./result_features/{method}/{data_type}", exist_ok=True)

augmentations = [
    "Default",
    "Mixup", 
    # "Local-FOMA", 
    "ES-Mixup",
    "Mixup-FOMA",
    # "Mixup-FOMA2",
]

# --- 1. データセットの準備 (ループの外で一度だけ行う) ---
if data_type == "stl10":
    num_classes = 10
    batch_size  = 64
    transform   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_dataset = STL10(root="./data", split="train", download=True, transform=transform)

elif data_type == "cifar100":
    epochs      = 250
    num_classes = 100
    batch_size  = 128
    transform   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_dataset = CIFAR100(root="./data", train=False, transform=transform, download=True)
    
    REPRESENTATIVE_20_FINE = [0, 1, 3, 8, 9, 15, 22, 23, 33, 37,
                              45, 46, 47, 49, 58, 59, 61, 71, 84, 95]
    REPRESENTATIVE_20_NAMES = [
        "apple","aquarium_fish","bear","bicycle","bottle","camel","clock","cloud","forest","house",
        "lawn_mower","leopard","man","orange","pickup_truck","plain","rocket","sea","table","whale"
    ]
    FINE_TO_20IDX = {fine:i for i, fine in enumerate(REPRESENTATIVE_20_FINE)}
    
    indices = [idx for idx, fine in enumerate(full_dataset.targets) if fine in REPRESENTATIVE_20_FINE]
    test_dataset = Subset(full_dataset, indices)

elif data_type == "cifar10":
    epochs      = 250
    num_classes = 10
    batch_size  = 128
    transform   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# --- 2. 全手法の特徴量抽出と次元圧縮を行う ---
results = {} # 結果を保存する辞書

print(f"Processing augmentations: {augmentations}")

for augment in augmentations:
    print(f"--- Processing {augment} ---")
    features_list = []
    labels_list = []
    
    if augment == "Mixup-FOMA" or augment == "Mixup-FOMA2":
        model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}_{k_foma}.pth"
    else:    
        model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"
    
    # モデルの準備
    if model_type == "resnet18":
        model = ResNet18().to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
    
    # 重みのロード
    try:
        checkpoint = torch.load(model_save_path, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Warning: Model file not found for {augment} at {model_save_path}. Skipping.")
        continue

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False, desc=f"Extracting {augment}"):
            images = images.to(device)
            features = model.extract_features(images)
            features_list.append(features.cpu())
            labels_list.append(labels)
    
    X = torch.cat(features_list, dim=0).numpy()
    y = torch.cat(labels_list, dim=0).numpy()

    if data_type == "cifar100":
        y = np.vectorize(FINE_TO_20IDX.get)(y)

    # 次元圧縮 (t-SNE / UMAP)
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap":
        reducer = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
    
    print(f"Running {method} for {augment}...")
    X_2d = reducer.fit_transform(X)
    
    # 結果を辞書に保存
    results[augment] = {
        "X_2d": X_2d,
        "y": y
    }

if not results:
    print("No results to plot.")
    exit()

# --- 3. 全手法を通じた軸の最大・最小 (目盛り) を計算 ---
all_x = np.concatenate([res["X_2d"][:, 0] for res in results.values()])
all_y = np.concatenate([res["X_2d"][:, 1] for res in results.values()])

x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()

# 余白を少し持たせる (5%)
margin_x = (x_max - x_min) * 0.05
margin_y = (y_max - y_min) * 0.05
xlim = (x_min - margin_x, x_max + margin_x)
ylim = (y_min - margin_y, y_max + margin_y)

print(f"Global limits determined: X={xlim}, Y={ylim}")

# --- 4. 固定された目盛りでプロット ---
for augment, data in results.items():
    X_2d = data["X_2d"]
    y = data["y"]

    plt.figure(figsize=(8, 6))
    
    # 表示タイトル等の調整
    plot_title_augment = "Baseline" if augment == "Default" else augment
    
    if data_type == "cifar100":
        cmap = plt.get_cmap('tab20', 20)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap, s=10, alpha=0.7)
        cbar = plt.colorbar(scatter, ticks=np.arange(20))
        cbar.set_label("Representative class (20)")
        try:
            cbar.ax.set_yticklabels(REPRESENTATIVE_20_NAMES)
        except Exception:
            pass
        title = f"{plot_title_augment}"
        save_suffix = "20classes"
    else:
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=6, alpha=1)
        plt.colorbar(scatter, label="Class label")
        title = f"t-SNE of Feature Representations ({plot_title_augment})"
        save_suffix = ""

    # ここで目盛り (範囲) を固定
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    
    save_path = f"./result_features/{method}/{data_type}/_{model_type}_{augment}_{k_foma}_{save_suffix}_fixed_scale.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close() # メモリ解放