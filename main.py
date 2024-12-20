import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from src.models.mlp import MLP
from sklearn.metrics import accuracy_score

print(torch.cuda.is_available())
i = 0
data_type = 'cifar10'

def main():
    epochs = 100
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)

    if data_type == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  transform=transforms.ToTensor(), download=True)
        test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif data_type == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  transform=transforms.ToTensor(), download=True)
        test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    n_samples = len(train_dataset)
    n_train   = int(n_samples * 0.8)
    n_val     = n_samples - n_train
    train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    score = 0.
    history = {'loss': [], 'accuracy': [], 'val_loss':[], 'val_accuracy': []}

    # Model Train #
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = val(model, val_loader, criterion, device)

        if score <= val_acc:
            print('save param')
            score = val_acc
            torch.save(model.state_dict(), './logs/' + str(epochs) + '_' + str(i) + '.pth') 
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')

    with open('./history/' + str(epochs) + '_' + str(i) + '.pickle', mode='wb') as f: 
        pickle.dump(history, f)

    # Model Test #
    model.load_state_dict(torch.load('./logs/' + str(epochs) + '_' + str(i) + '.pth'))
    model.eval()
    test = {'acc': [], 'loss': []}
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
    test['acc'].append(test_acc)
    test['loss'].append(test_loss)
    with open('./history/' + str(epochs) + '_' + 'test' + str(i) + '.pickle', mode='wb') as f: 
        pickle.dump(test, f)



def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for (images, labels) in tqdm(train_loader, leave=False):
        # images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        preds = model(images)
        loss  = criterion(preds, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
        
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    return train_loss, train_acc


def val(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for (images, labels) in val_loader:
            # images,labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)  
            
            preds = model(images)
            loss = criterion(preds, labels)
            
            val_loss += loss.item()
            val_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
            
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
    return val_loss, val_acc
    
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (images, labels) in test_loader:
            # images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            preds = model(images)
            loss = criterion(preds, labels)
            
            test_loss += loss.item()
            test_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
            
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
    return test_loss, test_acc

if __name__ == '__main__':
    main()