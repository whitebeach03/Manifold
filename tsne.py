import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

# モデル定義のインポート（環境に合わせてパスを調整してください）
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet

def get_balanced_subset(dataset, samples_per_class, num_classes):
    """
    各クラスから指定枚数ずつランダムにサンプリングしてSubsetを作成する関数
    """
    targets = np.array(dataset.targets)
    selected_indices = []
    
    print(f"Sampling {samples_per_class} images per class...")
    for c in range(num_classes):
        # クラスcのインデックスを抽出
        indices_c = np.where(targets == c)[0]
        
        # ランダムに選択（非復元抽出）
        if len(indices_c) >= samples_per_class:
            chosen = np.random.choice(indices_c, samples_per_class, replace=False)
        else:
            # データが足りない場合は警告を出して全データ使用
            print(f"Warning: Class {c} has only {len(indices_c)} samples.")
            chosen = indices_c
            
        selected_indices.extend(chosen)
    
    return Subset(dataset, selected_indices)

def main():
    parser = argparse.ArgumentParser(description="Visualize t-SNE with Balanced Sampling")
    parser.add_argument("--data_type",  type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--samples",    type=int, default=0, help="Samples per class (Default: 50 for C100, 500 for C10)")
    parser.add_argument("--model_path", type=str, default="", help="Path to model checkpoint. If empty, use Raw Image.")
    parser.add_argument("--model_type", type=str, default="resnet18", choices=["resnet18", "wide_resnet_28_10"])
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # 設定
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # デフォルト枚数の設定
    if args.samples == 0:
        if args.data_type == "cifar100":
            args.samples = 50  # 50 * 100 = 5000
        else:
            args.samples = 500 # 500 * 10 = 5000

    print(f"Dataset: {args.data_type}")
    print(f"Samples per class: {args.samples}")
    print(f"Total samples: {args.samples * (100 if args.data_type=='cifar100' else 10)}")

    # データセットのロード
    if args.data_type == "cifar100":
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        num_classes = 100
        dataset_class = torchvision.datasets.CIFAR100
    else:
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        num_classes = 10
        dataset_class = torchvision.datasets.CIFAR10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # テストセットを使用（訓練セットの分布を見たい場合は train=True に変更してください）
    full_dataset = dataset_class(root="./data", train=False, transform=transform, download=True)
    
    # 均等サンプリング
    subset = get_balanced_subset(full_dataset, args.samples, num_classes)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=2)

    features_list = []
    labels_list = []

    # --- 特徴抽出フェーズ ---
    if args.model_path and os.path.exists(args.model_path):
        # A. 学習済みモデルの特徴空間を使用
        print(f"Loading Model from {args.model_path} ...")
        
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
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Extracting Features"):
                images = images.to(device)
                # extract_featuresメソッドがある前提。なければフックかforward修正が必要
                if hasattr(model, 'extract_features'):
                    feats = model.extract_features(images)
                else:
                    # ResNet18の簡易対応 (avgpoolまで通す)
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
        title_suffix = "(Feature Space)"
        
    else:
        # B. 生画像を使用（入力空間）
        print("Using Raw Images (Input Space) ...")
        for images, labels in tqdm(loader, desc="Loading Images"):
            # (B, C, H, W) -> (B, C*H*W)
            flat_imgs = images.view(images.size(0), -1).numpy()
            features_list.append(flat_imgs)
            labels_list.append(labels.numpy())
            
        X = np.concatenate(features_list, axis=0)
        
        # 生画像は次元が高すぎる(3072)ので、t-SNEの前にPCAで50次元程度に落とすのが定石
        print("Applying PCA (3072 -> 50) for better t-SNE results...")
        pca = PCA(n_components=50, random_state=args.seed)
        X = pca.fit_transform(X)
        title_suffix = "(Input Space / Raw Pixel)"

    y = np.concatenate(labels_list, axis=0)

    # --- t-SNE 実行 ---
    print(f"Running t-SNE on {X.shape} ...")
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    # --- プロット ---
    plt.figure(figsize=(10, 8))
    
    if args.data_type == "cifar100":
        # クラス数が多いので色は20色パレットを使い回すか、特定のcolormapを使う
        cmap = plt.get_cmap('tab20')
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=cmap, s=10, alpha=0.6)
        # CIFAR-100は凡例をつけると邪魔なのでつけない、または代表的なものだけにする
        plt.title(f"t-SNE of CIFAR-100 {title_suffix}\n(50 samples/class, Total 5000)")
        
    else:
        # CIFAR-10
        cmap = plt.get_cmap('tab10')
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=cmap, s=10, alpha=0.7)
        plt.colorbar(scatter, ticks=range(10), label="Class ID")
        plt.title(f"t-SNE of CIFAR-10 {title_suffix}\n(500 samples/class, Total 5000)")

    # --- 修正箇所: 軸を表示する ---
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3) # グリッド線もあった方が見やすい場合が多い
    # plt.axis('off') # コメントアウトして軸を表示
    # ---------------------------
    
    os.makedirs("./result_tsne_balanced", exist_ok=True)
    save_path = f"./result_tsne_balanced/{args.data_type}_{'model' if args.model_path else 'raw'}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    main()