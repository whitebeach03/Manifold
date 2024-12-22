from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import KernelPCA

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