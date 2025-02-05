import numpy as np
import matplotlib.pyplot as plt
import os

red = "umap"

X = np.load(f'./our_dataset/images_{red}.npy')  # Shape: (5000, 1, 96, 96)
y = np.load(f'./our_dataset/labels_{red}.npy')

# 保存フォルダ
output_dir = "class_images"
os.makedirs(output_dir, exist_ok=True)

# クラスごとに処理
num_classes = 10
num_images_per_class = 100
grid_size = (10, 10)  # 10行 × 5列

for class_label in range(num_classes):
    # クラスに属するインデックスを取得
    indices = np.where(y == class_label)[0]
    
    # 50枚をランダムに選択
    selected_indices = np.random.choice(indices, num_images_per_class, replace=False)
    
    # 画像を 10×5 のグリッドに配置
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(50, 50))
    
    for i, ax in enumerate(axes.flat):
        img = X[selected_indices[i], 0]  # (96, 96) のグレースケール画像
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    # 画像を保存
    save_path = os.path.join(output_dir, f"class_{class_label}_{red}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

print(f"各クラスの画像が {output_dir} フォルダに保存されました。")
