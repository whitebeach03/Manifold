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
from src.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

i = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_type",  type=str, default="stl10",    choices=["mnist", "cifar10", "stl10"])
    parser.add_argument("--model_type", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    args = parser.parse_args() 

    epochs     = args.epochs
    batch_size = args.batch_size
    data_type  = args.data_type
    model_type = args.model_type
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select Model 
    if model_type == 'resnet18':
        model = ResNet18().to(device)
    elif model_type == 'resnet34':
        model = ResNet34().to(device)
    elif model_type == 'resnet50':
        model = ResNet50().to(device)
    elif model_type == 'resnet101':
        model = ResNet101().to(device)
    elif model_type == 'resnet152':
        model = ResNet152().to(device)

    # Loading Dataset
    if data_type == 'mnist':
        transform     = transforms.ToTensor()
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  transform=transform, download=True)
        test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif data_type == 'cifar10':
        transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  transform=transform, download=True)
        test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    elif data_type == 'stl10':
        transform     = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        train_dataset = torchvision.datasets.STL10(root='./data', split='train', transform=transform, download=True)
        test_dataset  = torchvision.datasets.STL10(root='./data', split='test',  transform=transform, download=True)
    
    n_samples = len(train_dataset)
    n_train   = int(n_samples * 0.8)
    n_val     = n_samples - n_train

######################################################################################################################################################################

    train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
    
######################################################################################################################################################################

    # _, val_dataset = random_split(train_dataset, [n_train, n_val])

    # val_loader  = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # # 訓練データの読み込み（生成データ）
    # generated_data = np.load('all_generated_high_dim_data.npy')
    # labels = np.concatenate([np.full(500, i) for i in range(10)])  # 各クラス500サンプル

    # train_dataset_generated = GeneratedDataset(generated_data, labels)
    # train_loader = DataLoader(dataset=train_dataset_generated, batch_size=batch_size, shuffle=True)

######################################################################################################################################################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    score     = 0.0
    history   = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    os.makedirs(f'./logs/{model_type}',        exist_ok=True)
    os.makedirs(f'./history/{model_type}',     exist_ok=True)
    os.makedirs(f'./result_plot/{model_type}', exist_ok=True)

    # Train 
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = val(model, val_loader, criterion, device)

        if score <= val_acc:
            print('Save model parameters...')
            score = val_acc
            model_save_path = f'./logs/{model_type}/{data_type}_{epochs}.pth'
            torch.save(model.state_dict(), model_save_path)

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        print(f'Epoch: {epoch+1} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f}')

    with open(f'./history/{model_type}/{data_type}_{epochs}.pickle', 'wb') as f:
        pickle.dump(history, f)

    # Test 
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}')

    test_history = {'acc': test_acc, 'loss': test_loss}
    with open(f'./history/{model_type}/{data_type}_{epochs}_test.pickle', 'wb') as f:
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