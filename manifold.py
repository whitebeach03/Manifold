import argparse
import numpy as np
from sklearn.datasets import make_swiss_roll
from scipy.stats import multivariate_normal
from src.reducer import *
from src.regressor import *
from src.sampling import *
from src.plot_data import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--red', default='lle',   choices=['kpca', 'lle'])
    parser.add_argument('--reg', default='rf',    choices=['svr', 'rf'])
    parser.add_argument('--sam', default='mixup', choices=['kde', 'mixup'])
    args = parser.parse_args() 
    
    red = args.red
    reg = args.reg
    sam = args.sam
    
    ### Swiss Roll ###
    n_samples   = 5000
    noise       = 0.05
    data, color = make_swiss_roll(n_samples=n_samples, noise=noise)

    ### Dimensionality reduction ###
    if red == 'kpca':
        reduced_data, kpca_model = kernel_pca_reduction(data, kernel='rbf', n_components=2, gamma=0.01, random_state=42)
    elif red == 'lle':
        reduced_data, lle_model = lle_reduction(data, n_components=2, n_neighbors=10)

    ### Train Manifold Regressor ###
    if reg == 'svr':
        regressors = train_manifold_regressor(reduced_data, data, kernel='rbf', C=10.0, gamma=0.1)
    elif reg == 'rf':
        regressors = train_manifold_regressor_rf(reduced_data, data, n_estimators=100, max_depth=None)

    ### Generate Low-Dimensional Data ###
    if sam == 'kde':
        new_low_dim_data = generate_samples_from_kde(reduced_data, n_samples=5000)
    elif sam == 'mixup':
        new_low_dim_data = generate_samples_from_mixup(reduced_data, n_samples=5000)

    ### Generate High Dimensional Data using Regressor ###
    generated_high_dim_data = generate_high_dim_data(regressors, new_low_dim_data)

    ### Visualization ###
    plot_high_dim_comparison_with_overlay(data, color, generated_high_dim_data)
    plot_high_dim_comparison(data, color, generated_high_dim_data)
    plot_low_dim_comparison(reduced_data, new_low_dim_data)


def generate_high_dim_data(regressors, low_dim_data):
    high_dim_data = np.zeros((low_dim_data.shape[0], len(regressors)))
    for i, regressor in enumerate(regressors):
        high_dim_data[:, i] = regressor.predict(low_dim_data)
    return high_dim_data


if __name__ == "__main__":
    main()