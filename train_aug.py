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
from torchvision.datasets import STL10, CIFAR10
from torch.utils.data import DataLoader, random_split
from src.models.resnet import ResNet18

def main():
    data_type = "cifar10"
    
    if data_type == "stl10":
        epochs = 150
        batch_size = 64
        base_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        train_dataset = STL10(root="./data", split="test", download=True, transform=base_transform)
        test_dataset = STL10(root="./data", split="train", download=True, transform=base_transform)
        train_loader, val_loader = create_loaders(train_dataset, split_path='data_split_indices.pkl', batch_size=batch_size)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    elif data_type == "cifar10":
        N = 12500
        N_train = int(N * 0.8)
        N_train_per = int(N / 10)
        # 1000=1250(batch_size=32), 5000=6250(batch_size=64), 10000=12500(batch_size=128)
        epochs = 200
        batch_size = 128
        base_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = CIFAR10(root='./data', train=True,  transform=base_transform, download=True)
        # train_dataset = create_balanced_subset(train_dataset, num_classes=10, samples_per_class=N_train_per)
        test_dataset = CIFAR10(root='./data', train=False, transform=base_transform, download=True)
        # train_loader, val_loader = create_loaders(train_dataset, split_path=f'data_split_indices_cifar_{N_train}.pkl', batch_size=batch_size)
        n_samples = len(train_dataset)
        n_train   = int(n_samples * 0.8)
        n_val     = n_samples - n_train
        train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
        print(len(train_dataset), len(val_dataset))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データ拡張のリスト
    augmentations = {
        "Manifold-Mixup": transforms.Compose([base_transform]),
        # "Mixup": transforms.Compose([base_transform]),
        # "Original": transforms.Compose([base_transform]),
        # "Flipping": transforms.Compose([
        #     base_transform,
        #     transforms.RandomApply([transforms.RandomHorizontalFlip(p=1.0)], p=0.5)
        # ]),
        # "Cropping": transforms.Compose([
        #     base_transform,
        #     transforms.RandomApply([transforms.RandomResizedCrop(size=96, scale=(0.7, 1.0))], p=0.5)
        # ]),
        # "Rotation": transforms.Compose([
        #     base_transform,
        #     transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.5)
        # ]),
        # "Translation": transforms.Compose([
        #     base_transform,
        #     transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], p=0.5)
        # ]),
        # "Noisy": transforms.Compose([
        #     base_transform,
        #     transforms.RandomApply([transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))], p=0.5)
        # ]),
        # "Blurring": transforms.Compose([
        #     base_transform, 
        #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5)
        # ]),
        # "Random-Erasing": transforms.Compose([
        #     base_transform, 
        #     transforms.RandomApply([transforms.RandomErasing(p=1.0, scale=(0.1, 0.3), ratio=(0.3, 3.3))], p=0.5)
        # ])
    }
    
    for name, transform in augmentations.items():
        print(f"\n==> Training with {name} data augmentation...")
        
        model = ResNet18().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        score     = 0.0
        history   = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        os.makedirs(f'./logs/resnet18/{name}',    exist_ok=True)
        os.makedirs(f'./history/resnet18/{name}', exist_ok=True)
        
        # Train 
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, name, aug_ok=False, epochs=epoch)
            val_loss, val_acc     = val(model, val_loader, criterion, device, name, aug_ok=False)

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
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.eval()
        test_loss, test_acc = test(model, test_loader, criterion, device, name, aug_ok=False)
        print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}')

        test_history = {'acc': test_acc, 'loss': test_loss}
        with open(f'./history/resnet18/{name}/{data_type}_{epochs}_test.pickle', 'wb') as f:
            pickle.dump(test_history, f)
        

def train(model, train_loader, criterion, optimizer, device, augment, aug_ok, epochs):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0
    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        # if augment == "Mixup":
        #     images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
        #     preds = model(images, labels, device, augment, aug_ok)
        #     loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        # elif augment == "Manifold-Mixup":
        #     preds, y_a, y_b, lam = model(images, labels, device, augment, mixup_hidden=True)
        #     loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        # else:  
        #     preds = model(images, labels, device, augment, aug_ok)
        #     loss  = criterion(preds, labels)
        
        if epochs < 100:
            preds = model(images, labels, device, augment, aug_ok)
            loss  = criterion(preds, labels)
        else:
            preds, y_a, y_b, lam = model(images, labels, device, augment, mixup_hidden=True)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # if augment == "Mixup" or "Manifold-Mixup":
        #     train_acc += (lam * accuracy_score(y_a.cpu(), preds.argmax(dim=-1).cpu()) + (1 - lam) * accuracy_score(y_b.cpu(), preds.argmax(dim=-1).cpu()))
        # else:
        #     train_acc += accuracy_score(labels.cpu(), preds.argmax(dim=-1).cpu())
        
        if epochs < 100:
            train_acc += accuracy_score(labels.cpu(), preds.argmax(dim=-1).cpu())
        else:
            train_acc += (lam * accuracy_score(y_a.cpu(), preds.argmax(dim=-1).cpu()) + (1 - lam) * accuracy_score(y_b.cpu(), preds.argmax(dim=-1).cpu()))
        
    train_loss /= len(train_loader)
    train_acc  /= len(train_loader)
    return train_loss, train_acc

def val(model, val_loader, criterion, device, augment, aug_ok):
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images, labels, device, augment, aug_ok)
            loss  = criterion(preds, labels)

            val_loss += loss.item()
            val_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    val_loss /= len(val_loader)
    val_acc  /= len(val_loader)
    return val_loss, val_acc

def test(model, test_loader, criterion, device, augment, aug_ok):
    model.eval()
    test_loss = 0.0
    test_acc  = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images, labels, device, augment, aug_ok)
            loss  = criterion(preds, labels)

            test_loss += loss.item()
            test_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    test_loss /= len(test_loader)
    test_acc  /= len(test_loader)
    return test_loss, test_acc

def mixup_data(x, y, alpha=1.0, use_cuda=True):
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



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == "__main__":
    main()