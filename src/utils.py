import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate_regression(regressors, low_dim_data, high_dim_data, test_size=0.2, random_state=42):
    """
    Parameters:
    - regressors (function): 回帰モデルをトレーニングする関数
                             例: train_manifold_regressor_knn
    - low_dim_data (ndarray): 入力データ（低次元）
    - high_dim_data (ndarray): 出力データ（高次元）
    - test_size (float): テストデータの割合
    - random_state (int): データ分割の再現性を確保するためのランダムシード
    """
    # トレーニング用とテスト用にデータを分割
    low_dim_train, low_dim_test, high_dim_train, high_dim_test = train_test_split(
        low_dim_data, high_dim_data, test_size=test_size, random_state=random_state
    )
    
    # 回帰モデルをトレーニング
    trained_regressors = regressors(low_dim_train, high_dim_train)
    
    # 高次元の予測結果を格納する配列
    high_dim_pred = np.zeros_like(high_dim_test)
    
    # 各次元ごとに予測を行う
    for i, regressor in enumerate(trained_regressors):
        high_dim_pred[:, i] = regressor.predict(low_dim_test)
    
    # 評価指標を計算
    mae = mean_absolute_error(high_dim_test, high_dim_pred)
    mse = mean_squared_error(high_dim_test, high_dim_pred)
    r2 = r2_score(high_dim_test, high_dim_pred)
    
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "R2_Score": r2
    }
    
    # 結果を出力
    print(f"Mean Absolute Error (MAE): {metrics['MAE']}")
    print(f"Mean Squared Error (MSE): {metrics['MSE']}")
    print(f"R² Score: {metrics['R2_Score']}")
    
    return metrics


def make_helix(n_samples):
    n_samples = 5000
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = np.sin(t)
    y = np.cos(t)
    z = t
    data = np.vstack((x, y, z)).T
    color = t
    return data, color

def make_spiral(n_samples):
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t
    data = np.vstack((x, y, z)).T
    color = t
    return data, color

