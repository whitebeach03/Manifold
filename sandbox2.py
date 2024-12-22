import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVR
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.ensemble import RandomForestRegressor

def main():
    ### Generate Swiss Roll ###
    n_samples = 5000
    noise = 0.05
    data, color = make_swiss_roll(n_samples=n_samples, noise=noise)

    ### Dimensionality reduction using KPCA ###
    # reduced_data, kpca_model = kernel_pca_reduction(data, kernel='rbf', n_components=2, gamma=0.01, random_state=42)
    reduced_data, lle_model = lle_reduction(data, n_components=2, n_neighbors=10)

    ### Train Manifold Regressor ###
    # regressors = train_manifold_regressor(reduced_data, data, kernel='rbf', C=10.0, gamma=0.1)
    regressors = train_manifold_regressor_rf(reduced_data, data, n_estimators=100, max_depth=None)

    ### Generate Low-Dimensional Data using KDE ###
    # new_low_dim_data = generate_samples_from_kde(reduced_data, n_samples=5000)
    new_low_dim_data = generate_samples_from_mixup(reduced_data, n_samples=5000)

    ### Generate High Dimensional Data ###
    generated_high_dim_data = generate_high_dim_data(regressors, new_low_dim_data)

    ### Visualization ###
    # Compare original and generated data in high-dimensional space
    # plot_high_dim_comparison_with_overlay(data, color, generated_high_dim_data)
    plot_high_dim_comparison(data, color, generated_high_dim_data)

    # Compare low-dimensional data distribution
    plot_low_dim_comparison(reduced_data, new_low_dim_data)

def lle_reduction(data, n_components=2, n_neighbors=10):
    """
    局所線形埋め込み (LLE) を用いて次元削減を行う関数
    """
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
    reduced_data = lle.fit_transform(data)
    return reduced_data, lle

def kernel_pca_reduction(data, kernel='rbf', n_components=2, gamma=None, random_state=42):
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    reduced_data = kpca.fit_transform(data)
    return reduced_data, kpca

def train_manifold_regressor(low_dim_data, high_dim_data, kernel='rbf', C=1.0, epsilon=0.1, gamma=None):
    regressors = []
    for i in range(high_dim_data.shape[1]):
        svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        svr.fit(low_dim_data, high_dim_data[:, i])
        regressors.append(svr)
    return regressors

def train_manifold_regressor_rf(low_dim_data, high_dim_data, n_estimators=100, max_depth=None, random_state=42):
    """
    ランダムフォレスト回帰を使用して高次元への写像を学習
    """
    regressors = []
    for i in range(high_dim_data.shape[1]):
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf.fit(low_dim_data, high_dim_data[:, i])
        regressors.append(rf)
    return regressors

def generate_high_dim_data(regressors, low_dim_data):
    high_dim_data = np.zeros((low_dim_data.shape[0], len(regressors)))
    for i, regressor in enumerate(regressors):
        high_dim_data[:, i] = regressor.predict(low_dim_data)
    return high_dim_data

def plot_high_dim_comparison(original_data, original_color, generated_data):
    """
    元の高次元データと生成された高次元データの比較をプロット
    """
    fig = plt.figure(figsize=(12, 6))

    # 元のデータ
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], c=original_color, cmap=plt.cm.viridis, s=10)
    ax1.set_title("Original High-Dimensional Data")

    # 生成されたデータ
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], c='black', s=10, alpha=0.5)
    ax2.set_title("Generated High-Dimensional Data")

    plt.savefig("high_dim_comparison.png")
    plt.show()

def plot_high_dim_comparison_with_overlay(original_data, original_color, generated_data):
    """
    高次元データを元データと生成データで重ねてプロット
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 元の高次元データをプロット
    ax.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], 
               c=original_color, cmap=plt.cm.viridis, s=10, label="Original Data", alpha=0.5)

    # 生成された高次元データをプロット
    ax.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], 
               c='black', s=10, label="Generated Data", alpha=0.7)

    ax.set_title("Overlay of Original and Generated High-Dimensional Data")
    ax.legend()
    plt.savefig("high_dim_overlay_comparison.png")
    plt.show()

def plot_high_dim_comparison_side_by_side_synced(original_data, original_color, generated_data):
    """
    高次元データを元データと生成データで横並びにプロットし、同期して動かす
    """
    from mpl_toolkits.mplot3d import Axes3D  # 必要なモジュールをインポート

    fig = plt.figure(figsize=(16, 6))

    # 元の高次元データをプロット
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2],
                           c=original_color, cmap=plt.cm.viridis, s=10, alpha=0.5)
    ax1.set_title("Original High-Dimensional Data")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # 生成された高次元データをプロット
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2],
                           c='black', s=10, alpha=0.7)
    ax2.set_title("Generated High-Dimensional Data")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # 初期同期: 両方の視点を一致させる
    def on_move(event):
        if event.inaxes == ax1:
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            fig.canvas.draw_idle()
        elif event.inaxes == ax2:
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            fig.canvas.draw_idle()

    # イベントリスナーを登録
    fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.tight_layout()
    plt.savefig("high_dim_side_by_side_synced_comparison.png")
    plt.show()


def plot_low_dim_comparison(original_low_dim_data, generated_low_dim_data):
    """
    元の低次元データと生成された低次元データの分布を比較
    """
    plt.figure(figsize=(8, 6))

    # 元の低次元データ
    plt.scatter(original_low_dim_data[:, 0], original_low_dim_data[:, 1], label='Original Low-Dim Data', alpha=0.5)

    # 生成された低次元データ
    plt.scatter(generated_low_dim_data[:, 0], generated_low_dim_data[:, 1], label='Generated Low-Dim Data (KDE)', alpha=0.5, color='black')

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Low-Dimensional Data Comparison")
    plt.legend()
    plt.savefig("low_dim_comparison.png")
    plt.show()

def generate_samples_from_kde(low_dim_data, n_samples, bandwidth=0.1):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(low_dim_data)
    new_samples = kde.sample(n_samples)
    return new_samples

def generate_samples_from_mixup(low_dim_data, n_samples, alpha=0.2):
    """
    Mixupによる低次元データの生成
    """
    indices = np.random.randint(0, low_dim_data.shape[0], size=(n_samples, 2))
    lambda_ = np.random.beta(alpha, alpha, size=n_samples).reshape(-1, 1)
    new_samples = (1 - lambda_) * low_dim_data[indices[:, 0]] + lambda_ * low_dim_data[indices[:, 1]]
    return new_samples

if __name__ == "__main__":
    main()