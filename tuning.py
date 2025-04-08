import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.models.resnet import ResNet18
from src.utils import *
from sklearn.metrics import accuracy_score

def main():
    epochs     = 100
    data_type  = "cifar10"
    model_type = "resnet18"
    augment    = "pca"
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # N = [1250, 6250, 12500]
    
    # for i in N:
    #     N_train = int(i * 0.8)
    #     N_train_per = int(i / 10)
    # Select Model
    if model_type == 'resnet18':
        model = ResNet18().to(device)
        if data_type == "cifar10":
            checkpoint_path = f"./logs/resnet18/Original/cifar10_200.pth"
        elif data_type == "stl10":
            checkpoint_path = "./logs/resnet18/Original/stl10_150.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    elif model_type == 'resnet34':
        model = ResNet34().to(device)
    elif model_type == 'resnet50':
        model = ResNet50().to(device)
    elif model_type == 'resnet101':
        model = ResNet101().to(device)
    elif model_type == 'resnet152':
        model = ResNet152().to(device)
        
    # 2. 全結合層以外をフリーズ
    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = True
    
    # Loading Dataset
    if data_type == 'mnist':
        transform     = transforms.ToTensor()
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  transform=transform, download=True)
        test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif data_type == 'cifar10':
        batch_size    = 128
        transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  transform=transform, download=True)
        # train_dataset = create_balanced_subset(train_dataset, num_classes=10, samples_per_class=N_train_per)
        test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        train_loader, val_loader = create_loaders(train_dataset, split_path=f'data_split_indices_cifar.pkl', batch_size=batch_size, save_if_missing=False)
    elif data_type == 'stl10':
        batch_size    = 64
        transform     = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        train_dataset = torchvision.datasets.STL10(root='./data', split='test', transform=transform, download=True)
        test_dataset  = torchvision.datasets.STL10(root='./data', split='train',  transform=transform, download=True)
        train_loader, val_loader = create_loaders(train_dataset, split_path='data_split_indices.pkl', batch_size=batch_size, save_if_missing=False)
    
    # n_samples = len(train_dataset)
    # n_val     = int(n_samples * 0.375) # validation data: 3,000 pattern
    # n_train   = n_samples - n_val     # train data:       5,000 pattern

    # train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])
    # val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

    bce_loss = nn.BCELoss().cuda()
    softmax = nn.Softmax(dim=1).cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3  # ← SGDより少し小さめから始めるのが無難
    )
    score     = 0.0
    history   = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    os.makedirs(f'./logs/{model_type}/Fine-Tuning/{augment}', exist_ok=True)
    os.makedirs(f'./history/{model_type}/Fine-Tuning/{augment}', exist_ok=True)
    
    # Train 
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, augment)
        val_loss, val_acc     = val(model, val_loader, criterion, device, augment)

        if score <= val_acc:
            print('Save model parameters...')
            score = val_acc
            model_save_path = f'./logs/{model_type}/Fine-Tuning/{augment}/{data_type}_{epochs}.pth'
            torch.save(model.state_dict(), model_save_path)

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        print(f'| {epoch+1} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f} |')

    with open(f'./history/{model_type}/Fine-Tuning/{augment}/{data_type}_{epochs}.pickle', 'wb') as f:
        pickle.dump(history, f)

    # Test 
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()
    test_loss, test_acc = test(model, test_loader, criterion, device, augment)
    print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}')

    test_history = {'acc': test_acc, 'loss': test_loss}
    with open(f'./history/{model_type}/Fine-Tuning/{augment}/{data_type}_{epochs}_test.pickle', 'wb') as f:
        pickle.dump(test_history, f)

def train(model, train_loader, criterion, optimizer, device, augment):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0
    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        preds = model(images, labels, device, augment, aug_ok=True)
        loss  = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_score(labels.cpu(), preds.argmax(dim=-1).cpu())
            
    train_loss /= len(train_loader)
    train_acc  /= len(train_loader)
    return train_loss, train_acc

def val(model, val_loader, criterion, device, augment):
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images, labels, device, augment)
            loss  = criterion(preds, labels)

            val_loss += loss.item()
            val_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    val_loss /= len(val_loader)
    val_acc  /= len(val_loader)
    return val_loss, val_acc

def test(model, test_loader, criterion, device, augment):
    model.eval()
    test_loss = 0.0
    test_acc  = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images, labels, device, augment)
            loss  = criterion(preds, labels)

            test_loss += loss.item()
            test_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    test_loss /= len(test_loader)
    test_acc  /= len(test_loader)
    return test_loss, test_acc

if __name__ == "__main__":
    main()
        