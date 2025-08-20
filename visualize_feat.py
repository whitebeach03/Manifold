# import os
# import torch
# import torchvision
# import random
# import torchvision.transforms as transforms
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import argparse
# from torch.utils.data import DataLoader, Subset
# from src.models.resnet import ResNet18
# from sklearn.manifold import TSNE
# from tqdm import tqdm
# from src.models.wide_resnet import Wide_ResNet
# from torchvision.datasets import STL10, CIFAR10, CIFAR100
# from umap import UMAP

# method     = "tsne"

# parser = argparse.ArgumentParser()
# parser.add_argument("--i",          type=int, default=1)
# parser.add_argument("--epochs",     type=int, default=400)
# parser.add_argument("--data_type",  type=str, default="cifar100",  choices=["stl10", "cifar100", "cifar10"])
# parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
# parser.add_argument("--k_foma",     type=int, default=0)
# args = parser.parse_args() 

# i          = args.i
# epochs     = args.epochs
# data_type  = args.data_type
# model_type = args.model_type
# k_foma     = args.k_foma
# device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.makedirs(f"./result_features/{method}/{data_type}", exist_ok=True)

# augmentations = [
#     # "FOMA-Mixup",
#     # "Default",
#     # "Mixup", 
#     # "Local-FOMA", 
#     "Mixup-FOMA2",
# ]

# if data_type == "stl10":
#     num_classes = 10
#     batch_size  = 64
# elif data_type == "cifar100":
#     epochs      = 400
#     num_classes = 100
#     batch_size  = 128
# elif data_type == "cifar10":
#     epochs      = 250
#     num_classes = 10
#     batch_size  = 128

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
#         test_dataset  = CIFAR100(root="./data", train=False, transform=transform, download=True)
#     elif data_type == "cifar10":
#         transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#         test_dataset  = CIFAR10(root="./data", train=False, transform=transform, download=True)
    
#     # テストセットの1/10だけ使う
#     # total_len = len(test_dataset)
#     # subset_len = total_len // 10
#     # indices = random.sample(range(total_len), subset_len)
#     # test_dataset = Subset(test_dataset, indices)
    
#     test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

#     model.load_state_dict(torch.load(model_save_path, weights_only=True))
#     model.eval()
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, leave=False):
#             images = images.to(device)
#             features = model.extract_features(images)
#             features_list.append(features.cpu())
#             labels_list.append(labels)
#     X = torch.cat(features_list, dim=0).numpy()
#     y = torch.cat(labels_list, dim=0).numpy()

#     if method == "tsne":
#         reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1500)
#     elif method == "umap":
#         # reducer = UMAP(n_components=2, random_state=42)
#         reducer = UMAP(
#             n_components=2,
#             n_neighbors=15,
#             min_dist=0.1,
#             metric='euclidean',
#             random_state=42
#         )
#     X_2d = reducer.fit_transform(X)

#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=6, alpha=0.7)
#     plt.colorbar(scatter, label="Class label")
#     plt.title("t-SNE of ResNet Feature Representations")
#     plt.xlabel("Dim 1")
#     plt.ylabel("Dim 2")
#     plt.tight_layout()
#     plt.savefig(f"./result_features/{method}/{data_type}/{augment}.png")


#     # # カラーマップを100クラスに対応させる
#     # num_classes = 100  # ここはdata_typeに応じて変えるか、すでにある変数を使う
#     # cmap = cm.get_cmap('nipy_spectral', num_classes)  # 高識別性で100色対応

#     # plt.figure(figsize=(10, 8))
#     # scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap, s=6, alpha=0.7)

#     # # カラーバー（省略してもよい）
#     # cbar = plt.colorbar(scatter, ticks=np.linspace(0, num_classes, 11, endpoint=False))
#     # cbar.set_label("Class label")

#     # plt.title("t-SNE of ResNet Feature Representations")
#     # plt.xlabel("Dim 1")
#     # plt.ylabel("Dim 2")
#     # plt.tight_layout()
#     # plt.savefig(f"./result_tsne/{data_type}/{augment}.png")

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
parser.add_argument("--i",          type=int, default=1)
parser.add_argument("--epochs",     type=int, default=400)
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
    # "FOMA-Mixup",
    # "Default",
    # "Mixup", 
    # "Local-FOMA", 
    "Mixup-FOMA2",
]

if data_type == "stl10":
    num_classes = 10
    batch_size  = 64
elif data_type == "cifar100":
    epochs      = 400
    num_classes = 100  # ← 学習済みモデルは100クラス想定のまま
    batch_size  = 128
elif data_type == "cifar10":
    epochs      = 250
    num_classes = 10
    batch_size  = 128

# ===== 追加: CIFAR-100 可視化を「代表20クラス」のみに制限する設定 =====
# 各スーパークラスから1クラスずつ任意選択（fine label index）
# 参考: apple(0)=fruit_and_vegetables, aquarium_fish(1)=fish, bear(3)=large_carnivores, bicycle(8)=vehicles_1,
# bottle(9)=household_containers, camel(15)=large_omnivores_and_herbivores, clock(22)=household_electrical_devices,
# cloud(23)=natural, forest(33)=outdoor_scenes, house(37)=household_furniture,
# lawn_mower(45)=household_electrical_devices, leopard(46)=large_carnivores, man(47)=people,
# orange(49)=fruit_and_vegetables, pickup_truck(58)=vehicles_2, plain(59)=outdoor_scenes,
# rocket(61)=vehicles_2, sea(71)=natural, table(84)=household_furniture, whale(95)=aquatic_mammals
REPRESENTATIVE_20_FINE = [0, 1, 3, 8, 9, 15, 22, 23, 33, 37,
                          45, 46, 47, 49, 58, 59, 61, 71, 84, 95]
REPRESENTATIVE_20_NAMES = [
    "apple","aquarium_fish","bear","bicycle","bottle","camel","clock","cloud","forest","house",
    "lawn_mower","leopard","man","orange","pickup_truck","plain","rocket","sea","table","whale"
]
FINE_TO_20IDX = {fine:i for i, fine in enumerate(REPRESENTATIVE_20_FINE)}

for augment in augmentations:
    features_list = []
    labels_list = []
    if k_foma == 0:
        model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"
    else:    
        model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}_{k_foma}.pth"
    
    if model_type == "resnet18":
        model = ResNet18().to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
    
    if data_type == "stl10":
        transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_dataset  = STL10(root="./data", split="train", download=True, transform=transform)
    elif data_type == "cifar100":
        transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        full_dataset  = CIFAR100(root="./data", train=False, transform=transform, download=True)

        # --- 代表20クラスのみ抽出（Subset化） ---
        # full_dataset.targets は fine labels (0-99)
        indices = [idx for idx, fine in enumerate(full_dataset.targets) if fine in REPRESENTATIVE_20_FINE]
        test_dataset = Subset(full_dataset, indices)

        # 注意: モデルは100クラスのまま（feature抽出なのでOK）
    elif data_type == "cifar10":
        transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_dataset  = CIFAR10(root="./data", train=False, transform=transform, download=True)
    
    # テストセットの1/10だけ使う（必要なら）
    # total_len = len(test_dataset)
    # subset_len = total_len // 10
    # indices = random.sample(range(total_len), subset_len)
    # test_dataset = Subset(test_dataset, indices)
    
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False):
            images = images.to(device)
            features = model.extract_features(images)
            features_list.append(features.cpu())
            labels_list.append(labels)
    X = torch.cat(features_list, dim=0).numpy()
    y = torch.cat(labels_list, dim=0).numpy()

    # --- CIFAR-100 のときは 0..19 に再マッピング（色分けのため） ---
    if data_type == "cifar100":
        # y は fine label (0-99) のはずなので，代表20以外は存在しない前提
        y = np.vectorize(FINE_TO_20IDX.get)(y)

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1500)
    elif method == "umap":
        reducer = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    if data_type == "cifar100":
        cmap = cm.get_cmap('tab20', 20)
        # bright_colors = [
        #     "red", "blue", "green", "orange", "purple",
        #     "cyan", "magenta", "yellow", "brown", "lime",
        #     "pink", "teal", "navy", "gold", "darkred",
        #     "darkgreen", "darkblue", "darkorange", "darkviolet", "olive"
        # ]
        # cmap = ListedColormap(bright_colors)
        # cmap = plt.cm.get_cmap("gist_rainbow", 20)  # 20 クラス対応
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap, s=6, alpha=0.7)
        cbar = plt.colorbar(scatter, ticks=np.arange(20))
        cbar.set_label("Representative class (20)")
        # 余裕があればクラス名も表示（長い場合は省略されます）
        try:
            cbar.ax.set_yticklabels(REPRESENTATIVE_20_NAMES)
        except Exception:
            pass
        title = "t-SNE of Feature Representations (CIFAR-100: 20 selected classes)"
        save_suffix = "_20classes"
    else:
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=6, alpha=1)
        plt.colorbar(scatter, label="Class label")
        title = "t-SNE of ResNet Feature Representations"
        save_suffix = ""

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(f"./result_features/{method}/{data_type}/{augment}_{k_foma}_{save_suffix}.png")
