import warnings
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.decomposition import KernelPCA, PCA
from umap import UMAP
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

### LLE (Locally Linear Embedding) ###
def lle_reduction(data, n_components=2, n_neighbors=10, method='modified'):
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method=method)
    reduced_data = lle.fit_transform(data)
    return reduced_data, lle

### Kernel PCA ###
def kernel_pca_reduction(data, kernel='rbf', n_components=2, gamma=None, random_state=42):
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    reduced_data = kpca.fit_transform(data)
    return reduced_data, kpca

### UMAP ###
def umap_reduction(data, n_components=2, n_neighbors=15, min_dist=0.1, random_state=None, n_jobs=-1):
    """    
    Parameters:
        data (array-like): 高次元データ (n_samples, n_features)
        n_components (int): 低次元の次元数
        n_neighbors (int): 各点の近傍点数 (UMAPの局所性の尺度)
        min_dist (float): 低次元空間での点間距離の最小値
        random_state (int, optional): 再現性のためのランダムシード
    """
    umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reduced_data = umap_model.fit_transform(data)
    return reduced_data, umap_model

### t-SNE ###
def tsne_reduction(data, n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=1000, random_state=None):
    """
    Parameters:
        data (array-like): 高次元データ (n_samples, n_features)
        n_components (int): 低次元の次元数 (デフォルト: 2)
        perplexity (float): t-SNEの局所性を制御するパラメータ (デフォルト: 30.0)
        learning_rate (float): 最適化の学習率 (デフォルト: 200.0)
        n_iter (int): 最適化の反復回数 (デフォルト: 1000)
        random_state (int, optional): 再現性のためのランダムシード

    Returns:
        reduced_data (array): 次元削減されたデータ (n_samples, n_components)
        tsne (TSNE): トレーニング済みのt-SNEオブジェクト
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter, random_state=random_state)
    reduced_data = tsne.fit_transform(data)
    return reduced_data, tsne

### PCA ###
def pca_reduction(data, n_components=2, random_state=42):
    """
    Parameters:
        data (array-like): 高次元データ (n_samples, n_features)
        n_components (int): 低次元の次元数 (デフォルト: 2)
        random_state (int, optional): 再現性のためのランダムシード

    Returns:
        reduced_data (array): 次元削減されたデータ (n_samples, n_components)
        pca (PCA): トレーニング済みのPCAオブジェクト
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca