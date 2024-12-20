from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from src.models.mlp import *
from sklearn.metrics import accuracy_score

# PCA適用後のデータセットを定義
class PCADataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).float(), self.labels[idx]

def main():
    args = parse_args()
    epochs = args.epochs
    data_type = args.data_type
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Manifold_MLP().to(device)

    # データセット読み込み
    if data_type == 'mnist':
        transform = transforms.ToTensor()
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif data_type == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # PCAを事前に学習
    print("Applying PCA to training data...")
    train_images = train_dataset.data.reshape(len(train_dataset), -1).astype(np.float32) / 255.0
    train_labels = train_dataset.targets
    pca = PCA(n_components=512)
    train_images_pca = pca.fit_transform(train_images)

    # PCA適用データセット
    train_dataset_pca = PCADataset(train_images_pca, train_labels)

    # データローダー
    n_samples = len(train_dataset_pca)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train
    train_dataset_pca, val_dataset = torch.utils.data.random_split(train_dataset_pca, [n_train, n_val])

    train_loader = DataLoader(dataset=train_dataset_pca, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)  # 元の画像データ
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)  # 元の画像データ

    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # トレーニングと検証ループ
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val(model, val_loader, criterion, device)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

    # テスト
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # モデルの保存
    torch.save(model.state_dict(), './logs/best_model.pth')

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        preds = model.forward_fc6(images)  # PCAで64次元に圧縮済みデータ
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=1).cpu().tolist())

    return total_loss / len(train_loader), total_acc / len(train_loader)

def val(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)  # 1次元ベクトル化
            preds = model(images)  # 元の画像データ
            loss = criterion(preds, labels)

            total_loss += loss.item()
            total_acc += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=1).cpu().tolist())

    return total_loss / len(val_loader), total_acc / len(val_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)  # 1次元ベクトル化
            preds = model(images)  # 元の画像データ
            loss = criterion(preds, labels)

            total_loss += loss.item()
            total_acc += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=1).cpu().tolist())

    return total_loss / len(test_loader), total_acc / len(test_loader)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with PCA-processed data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--data_type", type=str, choices=["mnist", "cifar10"], default="cifar10", help="Dataset to use")
    return parser.parse_args()

if __name__ == '__main__':
    main()
