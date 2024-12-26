import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import defaultdict
from src.reducer import *
from src.regressor import *
from src.sampling import *
from src.plot_data import *
from src.utils import *
from src.models.mlp import MLP

def main():
    n_new_samples = 10000
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データロード
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                               transform=transforms.ToTensor())
    train_data = train_dataset.data.view(-1, 784).float() / 255.0
    train_labels = train_dataset.targets
    test_data = test_dataset.data.view(-1, 784).float() / 255.0
    test_labels = test_dataset.targets

    # 元データのみでの学習
    print("=== Training with Original Data Only ===")
    original_dataset = TensorDataset(train_data, train_labels)
    train_loader_original = DataLoader(original_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=128, shuffle=False)

    # モデル設定
    model_original = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_original.parameters())

    # 元データの学習
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model_original, train_loader_original, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/10, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # 元データのテスト
    test_loss_original, top1_error_original, top5_error_original = evaluate(model_original, test_loader, criterion, device)
    print(f"Original Data - Test Loss: {test_loss_original:.4f}, Top-1 Error: {top1_error_original:.4f}, Top-5 Error: {top5_error_original:.4f}")

    # データ拡張
    print("\n=== Training with Augmented Data ===")
    # クラスラベルごとにデータを分割
    train_data_by_class = split_data_by_class(train_data, train_labels)

    # データ拡張の準備
    augmented_data_list = []
    augmented_labels_list = []

    # データ拡張 (UMAP -> Regressor -> High-Dimensional Data)
    for i in range(10):
        print(f"Data Augmentation for Class {i}...")
        reduced_data, _ = umap_reduction(train_data_by_class[i], n_components=3, n_neighbors=15, min_dist=0.1)
        regressors = train_manifold_regressor_knn(reduced_data, train_data_by_class[i], n_neighbors=5, weights='uniform', algorithm='auto')
        new_low_dim_data = generate_samples_from_knn(reduced_data, n_samples=n_new_samples)
        generated_high_dim_data = generate_high_dim_data(regressors, new_low_dim_data)

        # 拡張データと対応するラベルをリストに追加
        augmented_data_list.append(torch.tensor(generated_high_dim_data, dtype=torch.float32))
        augmented_labels_list.append(torch.full((10000,), i))  # クラスiのラベルを生成

    # 拡張データを統合
    augmented_data = torch.cat(augmented_data_list, dim=0)
    augmented_labels = torch.cat(augmented_labels_list, dim=0)

    # 元データと拡張データを統合
    combined_data = torch.cat([train_data, augmented_data], dim=0)
    combined_labels = torch.cat([train_labels, augmented_labels], dim=0)

    # データセットとデータローダーを作成
    combined_dataset = TensorDataset(combined_data, combined_labels)
    train_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=128, shuffle=False)

    # モデル設定
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 学習ループ
    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # テスト
    test_loss, top1_error, top5_error = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Top-1 Error: {top1_error:.4f}, Top-5 Error: {top5_error:.4f}")


def calculate_topk_error(output, target, k=5):
    """
    Top-kエラー率を計算する関数
    """
    topk_preds = torch.topk(output, k, dim=1).indices
    correct = torch.any(topk_preds == target.view(-1, 1), dim=1)
    return 1.0 - correct.float().mean().item()

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += accuracy_score(labels.cpu().numpy(), preds.argmax(dim=1).cpu().numpy())
    return train_loss / len(train_loader), train_acc / len(train_loader)

def evaluate(model, loader, criterion, device, top_k=(1, 5)):
    model.eval()
    loss, top1_error, top5_error = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss += criterion(preds, labels).item()
            top1_error += calculate_topk_error(preds, labels, k=1)
            top5_error += calculate_topk_error(preds, labels, k=5)
    return loss / len(loader), top1_error / len(loader), top5_error / len(loader)
    
def split_data_by_class(data, labels):
    class_data = defaultdict(list)
    for i, label in enumerate(labels):
        class_data[label.item()].append(data[i].numpy())  # データをクラスラベルごとに格納
    # Tensorに変換
    for label in class_data:
        class_data[label] = torch.tensor(class_data[label], dtype=torch.float32)
    return class_data


if __name__ == "__main__":
    main()
