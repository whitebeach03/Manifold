import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from src.utils import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split
from src.models.resnet import ResNet18

def main():
    epochs = 200
    batch_size = 128
    data_type = "stl10"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    
    # データ拡張のリスト
    augmentations = {
        "Original": transforms.Compose([base_transform]),
        "Flipping": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.RandomHorizontalFlip(p=1.0)], p=0.5)
        ]),
        "Cropping": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.RandomResizedCrop(size=96, scale=(0.7, 1.0))], p=0.5)
        ]),
        "Rotation": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.5)
        ]),
        "Translation": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], p=0.5)
        ]),
        "Noisy": transforms.Compose([
            base_transform,
            transforms.RandomApply([transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))], p=0.5)
        ]),
        "Blurring": transforms.Compose([
            base_transform, 
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5)
        ]),
        "Random-Erasing": transforms.Compose([
            base_transform, 
            transforms.RandomApply([transforms.RandomErasing(p=1.0, scale=(0.1, 0.3), ratio=(0.3, 3.3), value="random")], p=0.5)
        ])
    }
    
    # テストデータ / 検証データ（共通の変換を適用）
    test_dataset = STL10(root="./data", split="test", download=True, transform=base_transform)
    n_samples = len(test_dataset)
    n_val     = int(n_samples * 0.125) # validation data: 1,000 pattern
    n_test    = n_samples - n_val      # test data:       7,000 pattern
    test_dataset, val_dataset = random_split(test_dataset, [n_test, n_val])
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    for name, transform in augmentations.items():
        print(f"\n==> Training with {name} data augmentation...")
        
        # 学習データ
        train_dataset = STL10(root="./data", split="train", download=True, transform=transform)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        model = ResNet18().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        score     = 0.0
        history   = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        os.makedirs(f'./logs/resnet18/{name}',    exist_ok=True)
        os.makedirs(f'./history/resnet18/{name}', exist_ok=True)
        
        # Train 
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc     = val(model, val_loader, criterion, device)

            if score <= val_acc:
                print('Save model parameters...')
                score = val_acc
                model_save_path = f'./logs/resnet18/{name}/{data_type}_{epochs}.pth'
                torch.save(model.state_dict(), model_save_path)

            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            print(f'| {epoch+1} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f} |')

        with open(f'./history/resnet18/{name}/{data_type}_{epochs}.pickle', 'wb') as f:
            pickle.dump(history, f)
            
        # Test 
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}')

        test_history = {'acc': test_acc, 'loss': test_loss}
        with open(f'./history/resnet18/{name}/{data_type}_{epochs}_test.pickle', 'wb') as f:
            pickle.dump(test_history, f)
        

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0
    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        preds = model(images)
        loss  = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc  += accuracy_score(labels.cpu(), preds.argmax(dim=-1).cpu())
        
    train_loss /= len(train_loader)
    train_acc  /= len(train_loader)
    return train_loss, train_acc

def val(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images)
            loss  = criterion(preds, labels)

            val_loss += loss.item()
            val_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    val_loss /= len(val_loader)
    val_acc  /= len(val_loader)
    return val_loss, val_acc

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_acc  = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images)
            loss  = criterion(preds, labels)

            test_loss += loss.item()
            test_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    test_loss /= len(test_loader)
    test_acc  /= len(test_loader)
    return test_loss, test_acc

if __name__ == "__main__":
    main()