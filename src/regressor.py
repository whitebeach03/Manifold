from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

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