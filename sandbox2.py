import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def organize_by_class(dataset):
    class_data = {label: [] for label in range(10)}  
    for image, label in dataset:
        flattened_image = image.view(-1).numpy()
        class_data[label].append(flattened_image)
    for label in class_data:
        class_data[label] = np.array(class_data[label])
    return class_data

def manifold_perturbation(data, k=10, noise_scale=0.1):
    if isinstance(data, torch.Tensor):
        data_np = data.numpy()  # PyTorch → NumPy
    else:
        data_np = data

    N, D = data_np.shape  # (N, 9216)
    augmented_data = np.copy(data_np)
    perturbation_arr = []
    
    # 最近傍探索
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data_np)
    distances, indices = nbrs.kneighbors(data_np)
    
    for i in range(N):
        # 近傍点を取得
        neighbors = data_np[indices[i]]
        
        # PCAで局所的な主成分を取得
        pca = PCA(n_components=min(D, k))
        pca.fit(neighbors)
        principal_components = pca.components_  # 主成分方向
        
        # 主成分に沿ったランダムノイズを追加
        noise = np.random.normal(scale=noise_scale, size=(k,))
        perturbation = np.dot(principal_components.T, noise)
        perturbation_arr.append(perturbation)
        
        # データ点を摂動
        augmented_data[i] += perturbation

    # クリッピング（0-1の範囲を維持）
    augmented_data = np.clip(augmented_data, 0, 1)
    
    return torch.tensor(augmented_data, dtype=torch.float32), torch.tensor(perturbation_arr, dtype=torch.float32)  # NumPy → PyTorchに戻す

def display_augmented_images(label, data, name, num_images=100, grid_size=(10, 10)):
    if isinstance(data, torch.Tensor):
        data = data.numpy()  # PyTorch Tensor → NumPy
    
    num_images = min(num_images, data.shape[0])  # 画像数制限
    img_size = (96, 96)  # 画像サイズ
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            break
        img = data[i].reshape(img_size)  # 9216次元 → 96x96画像
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.savefig(f"{name}_{label}.png")
    plt.tight_layout()

# 使用例
if __name__ == "__main__":
    # ダミー画像データの作成（96x96グレースケール画像）
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor() 
    ])
    dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    data_by_class = organize_by_class(dataset)
    
    all_generated_high_dim_data = []
    all_labels = []
    
    for i in range(10):
        print("Label: ", i)
        data = data_by_class[i]
        # augmented_data, perturbation = manifold_perturbation(data, k=10, noise_scale=5.0)
        display_augmented_images(i, data, name="sample")
        # display_augmented_images(i, perturbation, name="perturbation")
        
        # データをリストに保存
    #     all_generated_high_dim_data.append(augmented_data)
        
    #     # ラベルをリストに保存
    #     labels = [i] * augmented_data.shape[0]  # ラベル l をデータ数分作成
    #     all_labels.extend(labels)
        
    #  # 全クラスのデータを結合
    # all_generated_high_dim_data = np.vstack(all_generated_high_dim_data)
    # N = all_generated_high_dim_data.shape[0] 
    # all_generated_high_dim_data = all_generated_high_dim_data.reshape(N, 1, 96, 96)
    # all_labels = np.array(all_labels)  # ラベルをNumPy配列に変換

    # # データとラベルを保存
    # np.save(f'./our_dataset/images_sample.npy', all_generated_high_dim_data)
    # np.save(f'./our_dataset/labels_sample.npy', all_labels)