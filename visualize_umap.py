import torch
import torchvision
import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet
from sklearn.manifold import TSNE  # kept for reference or removal
from umap import UMAP
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from tqdm import tqdm

# 設定
epochs     = 400
data_type  = "cifar100"
model_type = "wide_resnet_28_10"
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

augmentations = [
    # "Local-FOMA",
    # "Mixup",
    "Default",
]

for augment in augmentations:
    # 特徴量・ラベル 初期化
    features_list = []
    labels_list   = []

    model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_0.pth"

    # クラス数・バッチサイズ設定
    if data_type == "stl10":
        num_classes = 10
        batch_size  = 64
    elif data_type == "cifar100":
        num_classes = 100
        batch_size  = 128
    elif data_type == "cifar10":
        num_classes = 10
        batch_size  = 128

    # モデル構築
    if model_type == "resnet18":
        model = ResNet18().to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)

    # データセット準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if data_type == "stl10":
        test_dataset = STL10(root="./data", split="train", download=True, transform=transform)
    elif data_type == "cifar100":
        test_dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif data_type == "cifar10":
        test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    # テストセットの半分だけ使う
    total_len = len(test_dataset)
    subset_len = total_len // 2
    indices = random.sample(range(total_len), subset_len)
    test_dataset = Subset(test_dataset, indices)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # モデルロード
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()

    # 特徴量抽出
    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False):
            images = images.to(device)
            features = model.extract_features(images)
            features_list.append(features.cpu())
            labels_list.append(labels)

    # NumPy 変換
    X = torch.cat(features_list, dim=0).numpy()
    y = torch.cat(labels_list, dim=0).numpy()

    # UMAP で次元削減
    reducer = UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X)

    # プロット
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=6, alpha=0.7)
    plt.colorbar(scatter, label="Class label")
    plt.title("UMAP of ResNet Feature Representations")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.tight_layout()
    plt.savefig(f"./umap/{data_type}_{augment}_{epochs}.png")
    plt.close()
