import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVR
from scipy.stats import multivariate_normal

def main():
    ### Generate Swiss Roll ###
    n_samples   = 1500
    noise       = 0.05 
    data, color = make_swiss_roll(n_samples=n_samples, noise=noise)

    ### Dimensionality reduction using KPCA ###
    reduced_data, kpca_model = kernel_pca_reduction(data, kernel='rbf', n_components=2, gamma=0.01, random_state=42)
    print(reduced_data.shape)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label='Original Data', alpha=0.7)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()




    ### Plot Swiss Roll ###
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.viridis, s=10)
    # ax.set_title("3D Swiss Roll")
    # plt.savefig("swiss_roll.png")

def kernel_pca_reduction(data, kernel='rbf', n_components=2, gamma=None, random_state=42):
    """
    高次元データを非線形次元削減する関数
    """
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    reduced_data = kpca.fit_transform(data)
    return reduced_data, kpca

def train_manifold_regressor(low_dim_data, high_dim_data, kernel='rbf', C=1.0, epsilon=0.1, gamma=None):
    """
    低次元データから高次元データへの写像を学習する回帰モデルを構築
    """
    regressors = []
    for i in range(high_dim_data.shape[1]):
        svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        svr.fit(low_dim_data, high_dim_data[:, i])
        regressors.append(svr)
    return regressors

def generate_high_dim_data(regressors, low_dim_data):
    """
    低次元データから高次元データを生成
    """
    high_dim_data = np.zeros((low_dim_data.shape[0], len(regressors)))
    for i, regressor in enumerate(regressors):
        high_dim_data[:, i] = regressor.predict(low_dim_data)
    return high_dim_data

def sample_from_gaussian(mean, covariance, n_samples):
    """
    ガウシアン分布から低次元データをサンプリング
    """
    return multivariate_normal.rvs(mean=mean, cov=covariance, size=n_samples)

if __name__ == "__main__":
    main()

    