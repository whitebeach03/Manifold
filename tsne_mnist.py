import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src.methods.foma import foma

def main():
    data_type = "fo"

    # 1. データセット準備
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    # 2. クラスごとサブセット収集
    class_indices = {d: [] for d in range(10)}
    for idx, (_, lbl) in enumerate(mnist_train):
        if len(class_indices[lbl]) < 1024:
            class_indices[lbl].append(idx)
        if all(len(v) >= 1024 for v in class_indices.values()):
            break

    # 3. t-SNE 埋め込み & プロット設定
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for digit in range(10):
        subset = Subset(mnist_train, class_indices[digit])
        loader = DataLoader(subset, batch_size=1024, shuffle=False)
        
        feats = []
        for imgs, labels in loader:
            if data_type == "foma":
                imgs, _ = foma(imgs, labels, num_classes=10, alpha=1.0, rho=0.9)
            feats.append(imgs.view(imgs.size(0), -1))
        feats = torch.cat(feats, dim=0).numpy()
        
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            random_state=42,
            verbose=1
        )
        embedding = tsne.fit_transform(feats)
        
        ax = axes[digit]
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.6)
        ax.set_title(f'Digit {digit}')
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle('t-SNE projection for each MNIST class', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = "FOMA_tSNE.png" if data_type=="foma" else "TSNE_IMAGES.png"
    plt.savefig(fname)
    plt.show()

if __name__ == '__main__':
    main()
