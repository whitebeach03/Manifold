import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from tqdm import tqdm
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet

def compute_knn_distance(features, labels, k=5):
    """
    クラスごとにk近傍距離を計算して平均する
    """
    classes = np.unique(labels)
    total_dist = 0.0
    total_samples = 0
    
    # 全特徴量を正規化（重要！これでスケールの違いを排除）
    features = F.normalize(torch.from_numpy(features), p=2, dim=1)
    
    for c in tqdm(classes, desc="Calculating Density"):
        # クラスcのデータのみ抽出
        mask = (labels == c)
        feats_c = features[mask] # (N_c, Dim)
        num_c = feats_c.shape[0]
        
        if num_c <= k:
            continue
            
        # 距離行列を計算 (Euclidean Distance on Hypersphere)
        # dist = ||a - b||
        # メモリ節約のためループか、データ数が少なければ一括
        # ここではcdist的な計算
        dists = torch.cdist(feats_c, feats_c, p=2) # (N_c, N_c)
        
        # 昇順ソートしてk番目を取得 (0番目は自分自身なので距離0)
        # topkは大きい順なので、sortを使う
        sorted_dists, _ = torch.sort(dists, dim=1)
        
        # k番目の近傍までの距離 (インデックスk。0始まりなのでkでOK)
        # 例: k=1なら最も近い他人
        knn_dists = sorted_dists[:, k] 
        
        total_dist += knn_dists.sum().item()
        total_samples += num_c
        
    avg_knn_dist = total_dist / total_samples
    return avg_knn_dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="wide_resnet_28_10")
    parser.add_argument("--k", type=int, default=8, help="Measure distance to k-th neighbor")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # データセット設定
    if args.data_type == "cifar100":
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        num_classes = 100
        dset_class = CIFAR100
    else:
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        num_classes = 10
        dset_class = CIFAR10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # テストセットで評価（訓練セットでも可）
    dataset = dset_class(root="./data", train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    # モデルロード
    print(f"Loading {args.model_path} ...")
    if args.model_type == "resnet18":
        model = ResNet18().to(device)
    elif args.model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 特徴抽出
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting Features"):
            images = images.to(device)
            if hasattr(model, 'extract_features'):
                feats = model.extract_features(images)
            else:
                # ResNet18 fallback
                x = model.conv1(images)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                feats = x.view(x.size(0), -1)
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    # 密度計算
    print(f"\nComputing Average k-NN Distance (k={args.k}) ...")
    density_metric = compute_knn_distance(X, y, k=args.k)
    
    print("-" * 30)
    print(f"Dataset: {args.data_type}")
    print(f"Model  : {args.model_type}")
    print(f"Avg Distance to {args.k}-th Neighbor: {density_metric:.4f}")
    print("-" * 30)
    print("Value interpretation:")
    print("Lower = Denser (Tight clusters)")
    print("Higher = Sparser (Scattered points)")

if __name__ == "__main__":
    main()