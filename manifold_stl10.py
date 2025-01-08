import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torchvision
from sklearn.datasets import fetch_olivetti_faces
from src.reducer import *
from src.regressor import *
from src.sampling import *
from src.plot_data import *
from src.utils import *

def main():
    n_new_samples = 500
    all_generated_high_dim_data = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--red',  default='umap', choices=['kpca', 'lle', 'tsne', 'umap', 'pca'])
    parser.add_argument('--reg',  default='knn', choices=['svr', 'rf', 'gb', 'knn', 'poly'])
    parser.add_argument('--sam',  default='mixup', choices=['kde', 'mixup', 'knn'])
    args = parser.parse_args() 

    red = args.red
    reg = args.reg
    sam = args.sam
    # data_type = 'CIFAR10'
    data_type = 'STL10'

    ### Loading dataset ###
    print("Loading Dataset...")
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor() 
    ])

    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    test_dataset  = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)

    train_data_by_class = organize_by_class(train_dataset)
    test_data_by_class  = organize_by_class(test_dataset)

    ### Dimensionality reduction ###
    for l in range(10):
        data = train_data_by_class[l]
        print(data.shape)
        print("Dimensionality reduction...")
        if red == 'kpca':
            reduced_data, _ = kernel_pca_reduction(data, kernel='rbf', n_components=500, gamma=0.1, random_state=42)
        elif red == 'lle':
            reduced_data, _ = lle_reduction(data, n_components=3, n_neighbors=10, method='modified')
        elif red == 'tsne':
            reduced_data, _ = tsne_reduction(data, n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=1000, random_state=42)
        elif red == 'umap':
            reduced_data, _ = umap_reduction(data, n_components=3, n_neighbors=15, min_dist=0.1, random_state=None)
        elif red == 'pca':
            reduced_data, _ = pca_reduction(data, n_components=5000, random_state=42)
        plot_3d_data(reduced_data, color='blue', title=f"Low-Dimensional Data ({red})")
        # print(reduced_data.shape)
    
    ### Train Manifold Regressor ###
        print("Train Manifold Regressor...")
        if reg == 'svr':
            regressors = train_manifold_regressor(reduced_data, data, kernel='rbf', C=10.0, gamma=0.1)
        elif reg == 'rf':
            regressors = train_manifold_regressor_rf(reduced_data, data, n_estimators=100, max_depth=None)
        elif reg == 'gb':  
            regressors = train_manifold_regressor_gb(reduced_data, data, n_estimators=100, learning_rate=0.1, max_depth=3)
        elif reg == 'knn':
            regressors = train_manifold_regressor_knn(reduced_data, data, n_neighbors=5, weights='uniform', algorithm='auto')
        elif reg == 'poly':
            regressors = train_manifold_regressor_poly(reduced_data, data, degree=3)
    
    ### Generate Low-Dimensional Data ###
        print("Generate Low-Dimensional Data...")
        if sam == 'kde':
            new_low_dim_data = generate_samples_from_kde(reduced_data, n_samples=n_new_samples)
        elif sam == 'mixup':
            new_low_dim_data = generate_samples_from_mixup(reduced_data, n_samples=n_new_samples)
        elif sam == 'knn':
            new_low_dim_data = generate_samples_from_knn(reduced_data, n_samples=n_new_samples)
        plot_low_dim_3d(reduced_data, new_low_dim_data, red, reg, sam, data_type, l)
    
    ### Generate High Dimensional Data using Regressor ###
        print("Generate High-Dimensional Data using Regressor...")
        generated_high_dim_data = generate_high_dim_data(regressors, new_low_dim_data)
        show_images_together(data, generated_high_dim_data, num_images=10, l=l)
        print(generated_high_dim_data.shape)

        # リストに保存
        all_generated_high_dim_data.append(generated_high_dim_data)
    
    # 全クラスのデータを結合
    all_generated_high_dim_data = np.vstack(all_generated_high_dim_data)

    # 保存
    np.save('all_generated_high_dim_data.npy', all_generated_high_dim_data)


def generate_high_dim_data(regressors, low_dim_data):
    high_dim_data = np.zeros((low_dim_data.shape[0], len(regressors)))
    for i, regressor in enumerate(regressors):
        high_dim_data[:, i] = regressor.predict(low_dim_data)
    return high_dim_data

def normalize_image(image):
    """
    画像を[0, 1]の範囲に正規化する関数。
    """
    return np.clip(image, 0, 1)

def show_images_together(original_data, reconstructed_data, num_images=10, l=0):
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 5))

    for i in range(num_images):
        # 元画像
        original_image = normalize_image(original_data[i].reshape(96, 96))
        axes[0, i].imshow(original_image, cmap="gray")
        axes[0, i].set_title(f"Original #{i}")
        axes[0, i].axis("off")

        # 再構成画像
        reconstructed_image = normalize_image(reconstructed_data[i].reshape(96, 96))
        axes[1, i].imshow(reconstructed_image, cmap="gray")
        axes[1, i].set_title(f"Reconstructed #{i}")
        axes[1, i].axis("off")

    # 全体のタイトル
    plt.suptitle("Original and Reconstructed Images", fontsize=16)
    plt.tight_layout()
    filename = f"result/STL10/class_{l}.png"
    plt.savefig(filename)
    # plt.show()

def organize_by_class(dataset):
    class_data = {label: [] for label in range(10)}  
    for image, label in dataset:
        flattened_image = image.view(-1).numpy()
        class_data[label].append(flattened_image)
    for label in class_data:
        class_data[label] = np.array(class_data[label])
    return class_data
    
def plot_low_dim_3d(original_low_dim_data, generated_low_dim_data, red, reg, sam, data_type, l):
    """
    3次元データの比較プロットを表示
    - 元データと生成データのオーバーレイを3Dプロット
    - プロットの外観と可読性を向上
    """
    from mpl_toolkits.mplot3d import Axes3D
    import os

    # プロットの作成
    fig = plt.figure(figsize=(18, 12))

    # Overlayプロット
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(original_low_dim_data[:, 0], original_low_dim_data[:, 1], original_low_dim_data[:, 2], 
                c='blue', alpha=0.5, s=50, label='Original Low-Dim Data')
    ax.scatter(generated_low_dim_data[:, 0], generated_low_dim_data[:, 1], generated_low_dim_data[:, 2], 
                c='black', alpha=0.5, s=50, label='Generated Low-Dim Data')
    
    # タイトルとラベル
    ax.set_title("Overlay of Original and Generated Low-Dimensional Data", fontsize=16)
    ax.set_xlabel("Component 1", fontsize=14)
    ax.set_ylabel("Component 2", fontsize=14)
    ax.set_zlabel("Component 3", fontsize=14)
    ax.legend(fontsize=12)

    # グリッドと視点の設定
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.view_init(elev=30, azim=120)  # 視点を調整

    # 保存ディレクトリの作成
    output_dir = f"result/{data_type}/low_dim_3d"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{l}_{red}_{reg}_{sam}.png")

    # プロットの保存と表示
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    # plt.show()



if __name__ == "__main__":
    main()