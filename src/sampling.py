import numpy as np
import ot
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors

### カーネル密度推定 (Kernel Density Estimation) ###
def generate_samples_from_kde(low_dim_data, n_samples, bandwidth=0.1):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(low_dim_data)
    new_samples = kde.sample(n_samples)
    return new_samples

### Mixup ###
def generate_samples_from_mixup(low_dim_data, n_samples, alpha=0.2):
    indices = np.random.randint(0, low_dim_data.shape[0], size=(n_samples, 2))
    lambda_ = np.random.beta(alpha, alpha, size=n_samples).reshape(-1, 1)
    new_samples = (1 - lambda_) * low_dim_data[indices[:, 0]] + lambda_ * low_dim_data[indices[:, 1]]
    return new_samples

### k-Nearest Neighbors Sampling ###
def generate_samples_from_knn(low_dim_data, n_samples, n_neighbors=5):
    # Fit the nearest neighbors model to the data
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(low_dim_data)

    # Randomly sample indices from the dataset
    sampled_indices = np.random.randint(0, low_dim_data.shape[0], size=n_samples)

    # Find the nearest neighbors for each sampled index
    _, neighbors = nbrs.kneighbors(low_dim_data[sampled_indices])

    # Generate new samples by interpolating between a point and its neighbors
    new_samples = []
    for i, idx in enumerate(sampled_indices):
        # Choose a random neighbor for interpolation
        neighbor_idx = np.random.choice(neighbors[i])
        alpha = np.random.uniform(0, 1)  # Random weight for interpolation
        new_sample = (1 - alpha) * low_dim_data[idx] + alpha * low_dim_data[neighbor_idx]
        new_samples.append(new_sample)

    return np.array(new_samples)