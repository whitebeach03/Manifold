from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import KernelPCA
from umap import UMAP

def lle_reduction(data, n_components=2, n_neighbors=10):
    """
    局所線形埋め込み (LLE) を用いて次元削減を行う関数
    """
    # lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='modified')
    reduced_data = lle.fit_transform(data)
    return reduced_data, lle

def kernel_pca_reduction(data, kernel='rbf', n_components=2, gamma=None, random_state=42):
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    reduced_data = kpca.fit_transform(data)
    return reduced_data, kpca

def umap_reduction(data, n_components=2, n_neighbors=15, min_dist=0.1, random_state=None):
    """
    UMAPを用いて次元削減を行う関数
    
    Parameters:
        data (array-like): 高次元データ (n_samples, n_features)
        n_components (int): 低次元の次元数
        n_neighbors (int): 各点の近傍点数 (UMAPの局所性の尺度)
        min_dist (float): 低次元空間での点間距離の最小値
        random_state (int, optional): 再現性のためのランダムシード

    Returns:
        reduced_data (array-like): 次元削減後のデータ (n_samples, n_components)
        umap_model (UMAP object): 学習済みのUMAPモデル
    """
    umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reduced_data = umap_model.fit_transform(data)
    return reduced_data, umap_model