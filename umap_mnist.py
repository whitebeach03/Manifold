import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src.methods.foma import foma

def main():

    data_type = "foma"

    # 1. データセットの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    
    # 2. 各クラスごとにサブセット化
    class_indices = {digit: [] for digit in range(10)}
    for idx, (_, label) in enumerate(mnist_train):
        if len(class_indices[label]) < 1024:  # クラスごと最大1000サンプルまで
            class_indices[label].append(idx)
        # 全クラスが1000に達したらループ終了
        if all(len(v) >= 1024 for v in class_indices.values()):
            break

    # 3. UMAP で次元削減しつつプロット
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for digit in range(10):
        # サブセット抽出
        subset = Subset(mnist_train, class_indices[digit])
        loader = DataLoader(subset, batch_size=1024, shuffle=False)
        
        # 特徴（フラット画像）とラベル収集
        feats = []
        for imgs, labels in loader:
            if data_type == "foma":
                imgs, soft_labels = foma(imgs, labels, num_classes=10, alpha=1.0, rho=0.9) ## FOMA用に追加
            feats.append(imgs.view(imgs.size(0), -1))
        feats = torch.cat(feats, dim=0).numpy()
        
        # UMAP による 2D 埋め込み
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(feats)
        
        # プロット
        ax = axes[digit]
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.6)
        ax.set_title(f'Digit {digit}')
        ax.set_xticks([]); ax.set_yticks([])
    
    plt.suptitle('UMAP projection for each MNIST class', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if data_type == "foma":
        plt.savefig("FOMA_UMAP.png")
    else:
        plt.savefig("UMAP_IMAGES.png")
    plt.show()

if __name__ == '__main__':
    main()
