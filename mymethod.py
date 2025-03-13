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

def class_pca_augmentation(data, num_components=100, noise_scale=0.2):
    N, D = data.shape  # N: サンプル数, D: 特徴数（例: 9216）

    # データを NumPy に変換
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data

    # 平均を計算し、センタリング
    mean = np.mean(data_np, axis=0, keepdims=True)
    data_centered = data_np - mean

    # PCA の適用（クラス内で学習）
    pca = PCA(n_components=min(num_components, N))  # 主成分数はデータ数以下
    pca.fit(data_centered)
    principal_components = pca.components_  # (k, D)

    # 主成分空間に射影
    projected_data = np.dot(data_centered, principal_components.T)  # (N, k)

    # 主成分空間でノイズを加える
    noise = noise_scale * np.random.randn(*projected_data.shape)  # (N, k)
    projected_data_noisy = projected_data + noise

    # 元のデータ空間に戻す
    augmented_data = np.dot(projected_data_noisy, principal_components) + mean

    # PyTorch Tensor に変換
    return torch.tensor(augmented_data, dtype=torch.float32) if isinstance(data, torch.Tensor) else augmented_data


def display_augmented_images(label, data, num_components, num_images=100, grid_size=(10, 10)):
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
    plt.savefig(f"pca_aug/{num_components}/sample_{label}.png")
    plt.tight_layout()
    
def plot_explained_variance(data):
    # NumPy に変換
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data

    # PCA 適用（最大 D 次元）
    pca = PCA()
    pca.fit(data_np)

    # 寄与率と累積寄与率
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # プロット
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', markersize=1)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA: Cumulative Explained Variance")
    plt.grid()
    plt.savefig("variance.png")


# 使用例
if __name__ == "__main__":
    # ダミー画像データの作成（96x96グレースケール画像）
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor() 
    ])
    dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    data_by_class = organize_by_class(dataset)
    
    # plot_explained_variance(data_by_class[0])
    
    num_components = [100, 200, 300, 400, 500]
    
    for num_component in num_components:
        all_generated_high_dim_data = []
        all_labels = []
    
        for i in range(10):
            print("Label: ", i)
            data = data_by_class[i]
            # augmented_data = manifold_perturbation(data, k=10, noise_scale=5.0)
            augmented_data = class_pca_augmentation(data, num_components=num_component)
            display_augmented_images(i, augmented_data, num_components=num_component)
            
            # データをリストに保存
            all_generated_high_dim_data.append(augmented_data)
            
            # ラベルをリストに保存
            labels = [i] * augmented_data.shape[0]  # ラベル l をデータ数分作成
            all_labels.extend(labels)
            
        # 全クラスのデータを結合
        all_generated_high_dim_data = np.vstack(all_generated_high_dim_data)
        N = all_generated_high_dim_data.shape[0] 
        all_generated_high_dim_data = all_generated_high_dim_data.reshape(N, 1, 96, 96)
        all_labels = np.array(all_labels)  # ラベルをNumPy配列に変換

        # データとラベルを保存
        np.save(f'./our_dataset/images_sample_{num_component}.npy', all_generated_high_dim_data)
        np.save(f'./our_dataset/labels_sample_{num_component}.npy', all_labels)