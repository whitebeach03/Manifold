import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm
from src.models.mlp import MLP
from src.models.cnn import SimpleCNN
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

i = 0

def main():
    args = parse_args()
    epochs = args.epochs
    data_type = args.data_type
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)

    # # データセットの選択
    # if data_type == 'mnist':
    #     transform = transforms.ToTensor()
    #     train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    #     test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    # elif data_type == 'cifar10':
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    #     train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    #     test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    # elif data_type == 'stl10':
    #     transform = transforms.Compose([
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.ToTensor()
    #     ])
    #     train_dataset = torchvision.datasets.STL10(root='./data', split='train', transform=transform, download=True)
    #     test_dataset = torchvision.datasets.STL10(root='./data', split='test', transform=transform, download=True)
    
    # # トレーニング・検証データの分割
    # n_samples = len(train_dataset)
    # n_train = int(n_samples * 0.8)
    # n_val = n_samples - n_train
    # train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
######################################################################################################################################################################
    # 検証・テストデータの読み込み（元のSTL10データ）
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    train_dataset_original = torchvision.datasets.STL10(root='./data', split='train', transform=transform, download=True)
    test_dataset = torchvision.datasets.STL10(root='./data', split='test', transform=transform, download=True)

    # 検証データの分割
    n_samples = len(train_dataset_original)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train
    _, val_dataset = random_split(train_dataset_original, [n_train, n_val])

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 訓練データの読み込み（生成データ）
    generated_data = np.load('all_generated_high_dim_data.npy')
    labels = np.concatenate([np.full(500, i) for i in range(10)])  # 各クラス500サンプル

    train_dataset_generated = GeneratedDataset(generated_data, labels)
    train_loader = DataLoader(dataset=train_dataset_generated, batch_size=batch_size, shuffle=True)
######################################################################################################################################################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    score = 0.0
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    os.makedirs('./logs/normal', exist_ok=True)
    os.makedirs('./history/normal', exist_ok=True)

    # モデルのトレーニング
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val(model, val_loader, criterion, device)

        if score <= val_acc:
            print('Saving model parameters...')
            score = val_acc
            model_save_path = f'./logs/normal_{epochs}_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        print(f'Epoch: {epoch+1}, Loss: {train_loss:.3f}, Accuracy: {train_acc:.3f}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_acc:.3f}')

    with open(f'./history/normal_{epochs}.pickle', 'wb') as f:
        pickle.dump(history, f)

    # テストセットで評価
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}')

    test_history = {'acc': test_acc, 'loss': test_loss}
    with open(f'./history/normal_{epochs}_test.pickle', 'wb') as f:
        pickle.dump(test_history, f)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    return train_loss, train_acc


def val(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)

            val_loss += loss.item()
            val_acc += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)

            test_loss += loss.item()
            test_acc += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    return test_loss, test_acc


def parse_args():
    """コマンドライン引数を処理"""
    parser = argparse.ArgumentParser(description="Train a model with CIFAR-10 or MNIST")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--data_type", type=str, choices=["mnist", "cifar10", "stl10"], default="stl10", help="Dataset to use")
    return parser.parse_args()

class GeneratedDataset(Dataset):
    """生成データ用データセットクラス"""
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32).reshape(-1, 1, 96, 96)  # グレースケール画像の形状に変更
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == '__main__':
    main()