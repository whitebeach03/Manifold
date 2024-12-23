from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

### SVR ###
def train_manifold_regressor(low_dim_data, high_dim_data, kernel='rbf', C=1.0, epsilon=0.1, gamma=None):
    regressors = []
    for i in range(high_dim_data.shape[1]):
        svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        svr.fit(low_dim_data, high_dim_data[:, i])
        regressors.append(svr)
    return regressors

### Random Forest ###
def train_manifold_regressor_rf(low_dim_data, high_dim_data, n_estimators=100, max_depth=None, random_state=42):
    regressors = []
    for i in range(high_dim_data.shape[1]):
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf.fit(low_dim_data, high_dim_data[:, i])
        regressors.append(rf)
    return regressors

### Gradient Boosting ###
def train_manifold_regressor_gb(low_dim_data, high_dim_data, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    """
    Parameters:
        low_dim_data (array-like): 低次元データ
        high_dim_data (array-like): 高次元データ
        n_estimators (int): 決定木の数 (デフォルト: 100)
        learning_rate (float): 学習率 (デフォルト: 0.1)
        max_depth (int): 決定木の深さの最大値 (デフォルト: 3)
        random_state (int): ランダムシード (デフォルト: 42)

    Returns:
        list: 高次元データの各次元に対応するトレーニング済みの勾配ブースティング回帰モデルのリスト
    """
    regressors = []
    for i in range(high_dim_data.shape[1]):
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        gb.fit(low_dim_data, high_dim_data[:, i])
        regressors.append(gb)
    return regressors

### k-Nearest Neighbors ###
def train_manifold_regressor_knn(low_dim_data, high_dim_data, n_neighbors=5, weights='uniform', algorithm='auto'):
    """
    Parameters:
        low_dim_data (array-like): 低次元データ
        high_dim_data (array-like): 高次元データ
        n_neighbors (int): 近傍点の数 (デフォルト: 5)
        weights (str): 近傍点の重み ('uniform' または 'distance', デフォルト: 'uniform')
        algorithm (str): 近傍探索アルゴリズム ('auto', 'ball_tree', 'kd_tree', 'brute', デフォルト: 'auto')

    Returns:
        list: 高次元データの各次元に対応するトレーニング済みのk-近傍回帰モデルのリスト
    """
    regressors = []
    for i in range(high_dim_data.shape[1]):
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        knn.fit(low_dim_data, high_dim_data[:, i])
        regressors.append(knn)
    return regressors

### Polynomial Regression ###
def train_manifold_regressor_poly(low_dim_data, high_dim_data, degree=2):
    """
    Parameters:
        low_dim_data (array-like): 低次元データ
        high_dim_data (array-like): 高次元データ
        degree (int): 多項式の次数 (デフォルト: 2)

    Returns:
        list: 高次元データの各次元に対応するトレーニング済みの多項式回帰モデルのリスト
    """
    regressors = []
    for i in range(high_dim_data.shape[1]):
        model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear_reg', LinearRegression())
        ])
        model.fit(low_dim_data, high_dim_data[:, i])
        regressors.append(model)
    return regressors