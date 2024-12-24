import argparse
import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve
from scipy.stats import multivariate_normal
from src.reducer import *
from src.regressor import *
from src.sampling import *
from src.plot_data import *
from src.utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--red',  default='kpca', choices=['kpca', 'lle', 'tsne', 'umap'])
    parser.add_argument('--reg',  default='knn', choices=['svr', 'rf', 'gb', 'knn', 'poly'])
    parser.add_argument('--sam',  default='knn', choices=['kde', 'mixup', 'knn'])
    parser.add_argument('--data', default='spiral', choices=['swiss_roll', 's_curve', 'helix', 'spiral'])
    args = parser.parse_args() 
    
    red = args.red
    reg = args.reg
    sam = args.sam
    data_type = args.data
    
    ### Swiss Roll ###
    n_samples     = 5000
    n_new_samples = 5000
    noise         = 0.05
    # data, color   = make_swiss_roll(n_samples=n_samples, noise=noise)
    # data, color   = make_s_curve(n_samples=n_samples, noise=noise)
    # data, color   = make_helix(n_samples)
    # data, color   = make_spiral(n_samples)

    ### Dimensionality reduction ###
    print(f"Dimensionality reduction...")
    if red == 'kpca':
        reduced_data, _ = kernel_pca_reduction(data, kernel='rbf', n_components=2, gamma=0.01, random_state=42)
    elif red == 'lle':
        reduced_data, _ = lle_reduction(data, n_components=2, n_neighbors=10, method='modified')
    elif red == 'tsne':
        reduced_data, _ = tsne_reduction(data, n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=1000, random_state=42)
    elif red == 'umap':
        reduced_data, _ = umap_reduction(data, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

    ### Train Manifold Regressor ###
    print(f"Train Manifold Regressor...")
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
    print(f"Generate Low-Dimensional Data...")
    if sam == 'kde':
        new_low_dim_data = generate_samples_from_kde(reduced_data, n_samples=n_new_samples)
    elif sam == 'mixup':
        new_low_dim_data = generate_samples_from_mixup(reduced_data, n_samples=n_new_samples)
    elif sam == 'knn':
        new_low_dim_data = generate_samples_from_knn(reduced_data, n_samples=n_new_samples)
    plot_low_dim(reduced_data, new_low_dim_data, red, reg, sam, data_type)

    ### Generate High Dimensional Data using Regressor ###
    print(f"Generate High-Dimensional Data using Regressor...")
    generated_high_dim_data = generate_high_dim_data(regressors, new_low_dim_data)

    ### Visualization ###
    plot_high_dim_comparison(data, color, generated_high_dim_data, red, reg, sam, data_type)
    plot_high_dim_comparison_with_overlay(data, color, generated_high_dim_data, red, reg, sam, data_type)
    plot_high_dim(data, color, generated_high_dim_data, red, reg, sam, data_type)
    

def generate_high_dim_data(regressors, low_dim_data):
    high_dim_data = np.zeros((low_dim_data.shape[0], len(regressors)))
    for i, regressor in enumerate(regressors):
        high_dim_data[:, i] = regressor.predict(low_dim_data)
    return high_dim_data


if __name__ == "__main__":
    main()