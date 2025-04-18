import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def show_all_images(images, labels=None, indices=None, max_cols=8):
    num_images = len(images)
    cols = min(max_cols, num_images)
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        img = images[i]
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        title = ""
        if labels is not None:
            title += f"Label: {labels[i]}"
        if indices is not None:
            title += f"\nIdx: {indices[i]}"
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("never_correct_imgs.png")
    plt.show()




def main():
    # 読み込み
    images = np.load("never_correct_data/images.npy")   # shape: (N, H, W, C)
    labels = np.load("never_correct_data/labels.npy")   # shape: (N,)
    indices = np.load("never_correct_data/indices.npy") # shape: (N,)

    # 表示
    # 必要に応じて
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 全画像に適用
    unnorm_images = (images * std) + mean
    unnorm_images = np.clip(unnorm_images, 0, 1)  # 表示のためにクリップ

    # 表示
    show_all_images(unnorm_images, labels, indices)

if __name__ == "__main__":
    main()