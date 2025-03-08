import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def organize_by_class(dataset):
    class_data = {label: [] for label in range(10)}  
    for image, label in dataset:
        flattened_image = image.view(-1).numpy()
        class_data[label].append(flattened_image)
    for label in class_data:
        class_data[label] = np.array(class_data[label])
    return class_data

def display_augmented_images(label, data, num_images=100, grid_size=(10, 10)):
    """
    摂動後の画像を10x10グリッドで表示する。
    
    Parameters:
        data (torch.Tensor or np.ndarray): (N, 9216) の拡張データ
        num_images (int): 表示する画像の数（デフォルト100）
        grid_size (tuple): グリッドのサイズ（デフォルト 10x10）
    """
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
    plt.savefig(f"sample_{label}.png")
    plt.tight_layout()

def pca_based_augmentation(data, noise_scale=0.1, n_components=0.99, k=10):
    """
    PCA の主成分方向に沿ってデータ拡張を行う

    Parameters:
        data (numpy.ndarray or torch.Tensor): (N, D) のデータ行列
        noise_scale (float): 摂動の大きさ（加えるノイズのスケール）
        n_components (float or int): PCA の次元数（割合 or 固定値）
        k (int): 摂動を加える主成分の数

    Returns:
        numpy.ndarray or torch.Tensor: 摂動後のデータ (N, D)
    """

    # **PyTorch Tensor を NumPy に変換**
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()  # GPU → CPU 変換を含む
    else:
        data_np = data.copy()

    N, D = data_np.shape  # N: サンプル数, D: 特徴次元

    # **PCA を適用**
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_np)  # (N, d) に圧縮
    d = data_pca.shape[1]  # 圧縮後の次元数

    # **摂動を加える主成分の選択**
    k = min(k, d)  # `k` は `d` 以下に制限
    principal_components = pca.components_[:k]  # 上位 k 個の主成分ベクトル (k, D)

    # **各データに摂動を加えるためのランダムスケールを生成**
    noise_factors = np.random.normal(loc=0, scale=noise_scale, size=(N, k))  # (N, k)

    # **主成分方向に沿った摂動を適用**
    perturbation = np.dot(noise_factors, principal_components)  # (N, k) × (k, D) → (N, D)

    # **元のデータに摂動を加える**
    perturbed_data = data_np + perturbation

    # **データを [0,1] にクリップ**
    perturbed_data = np.clip(perturbed_data, 0, 1)

    # **PyTorch Tensor に戻す場合**
    return torch.tensor(perturbed_data, dtype=torch.float32) if isinstance(data, torch.Tensor) else perturbed_data


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
        print("LABEL: ", i)
        label = i
        data = data_by_class[i]
        data = pca_based_augmentation(data, noise_scale=10)
        display_augmented_images(label, data)
        
        
        
        
        # data = data_by_class[i]
        # augmented_data = manifold_perturbation(data, noise_scale=10)
        
        # # データをリストに保存
        # all_generated_high_dim_data.append(augmented_data)
        
        # # ラベルをリストに保存
        # labels = [i] * augmented_data.shape[0]  # ラベル l をデータ数分作成
        # all_labels.extend(labels)
        
     # 全クラスのデータを結合
    # all_generated_high_dim_data = np.vstack(all_generated_high_dim_data)
    # N = all_generated_high_dim_data.shape[0] 
    # all_generated_high_dim_data = all_generated_high_dim_data.reshape(N, 1, 96, 96)
    # all_labels = np.array(all_labels)  # ラベルをNumPy配列に変換

    # # データとラベルを保存
    # np.save(f'./our_dataset/images_sample.npy', all_generated_high_dim_data)
    # np.save(f'./our_dataset/labels_sample.npy', all_labels)