import numpy as np
from sklearn.neighbors import KernelDensity

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