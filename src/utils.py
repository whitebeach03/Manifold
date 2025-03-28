import numpy as np
import torch
import pickle
from torchvision import datasets, transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import random_split, Subset, DataLoader

def save_split_indices(dataset, val_ratio=0.375, split_path="data_split_indices.pkl", seed=42):
    """訓練・検証データの分割インデックスを生成・保存する（初回のみ使用）"""
    n_samples = len(dataset)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)

    with open(split_path, "wb") as f:
        pickle.dump((train_dataset.indices, val_dataset.indices), f)
    print(f"Saved split indices to {split_path}")


def load_split_datasets(dataset, split_path="data_split_indices.pkl"):
    """保存されたインデックスを使って Subset を生成"""
    with open(split_path, "rb") as f:
        train_indices, val_indices = pickle.load(f)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def create_loaders(dataset, split_path="data_split_indices.pkl", batch_size=64, num_workers=2, save_if_missing=True, val_ratio=0.375, seed=42):
    """データセットから train/val ローダーを作成。インデックスが無ければ保存。"""
    import os
    if not os.path.exists(split_path):
        if save_if_missing:
            save_split_indices(dataset, val_ratio=val_ratio, split_path=split_path, seed=seed)
        else:
            raise FileNotFoundError(f"Split file not found: {split_path}")

    train_dataset, val_dataset = load_split_datasets(dataset, split_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def evaluate_regression(regressors, low_dim_data, high_dim_data, test_size=0.2, random_state=42):
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


def load_mnist_data_by_label(n_samples_per_label=5000):
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    mnist_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    
    data_by_label = {label: [] for label in range(10)}
    
    for image, label in mnist_dataset:
        if len(data_by_label[label]) < n_samples_per_label:
            data_by_label[label].append(image.view(-1).numpy())  # Flatten the image
        
        if all(len(data_by_label[l]) >= n_samples_per_label for l in range(10)):
            break
    
    for label in data_by_label:
        data_by_label[label] = np.array(data_by_label[label])
    
    with open("mnist.pkl", "wb") as f:
        pickle.dump(data_by_label, f)
    
    return data_by_label

def generate_high_dim_data(regressors, low_dim_data):
    high_dim_data = np.zeros((low_dim_data.shape[0], len(regressors)))
    for i, regressor in enumerate(regressors):
        high_dim_data[:, i] = regressor.predict(low_dim_data)
    return high_dim_data

def organize_by_class(dataset):
    class_data = {label: [] for label in range(10)}  
    for image, label in dataset:
        flattened_image = image.view(-1).numpy()
        class_data[label].append(flattened_image)
    for label in class_data:
        class_data[label] = np.array(class_data[label])
    return class_data

def mixup_data_hidden(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)

def limit_dataset(dataset):
    # クラスごとに100枚ずつサンプリング
    num_classes = 10
    samples_per_class = 100

    # クラスごとのインデックスを取得
    class_indices = {i: [] for i in range(num_classes)}

    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)

    # 選択したインデックスをリスト化
    selected_indices = [idx for indices in class_indices.values() for idx in indices]

    # 新しいサブセットの作成（クラスごとに100枚）
    limited_train_dataset = Subset(dataset, selected_indices)
    return limited_train_dataset