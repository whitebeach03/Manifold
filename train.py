import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.cm as cm
import pickle
import argparse
import matplotlib.pyplot as plt
from src.utils import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet
from sklearn.manifold import TSNE

def main():
    for i in range(3):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=250)
        parser.add_argument("--data_type", type=str, default="cifar100", choices=["stl10", "cifar100", "cifar10"])
        parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "wide_resnet_28_10"])
        parser.add_argument("--alpha", type=float, default=1.0, help="MixUp interpolation coefficient (default: 1.0)")
        args = parser.parse_args() 

        epochs     = args.epochs
        data_type  = args.data_type
        model_type = args.model_type
        alpha      = args.alpha
        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Number of Classes & Batch Size
        if data_type == "stl10":
            num_classes = 10
            batch_size  = 64
        elif data_type == "cifar100":
            num_classes = 100
            batch_size  = 128
        elif data_type == "cifar10":
            num_classes = 10
            batch_size  = 128
        
        # Select Model
        if model_type == "resnet18":
            model = ResNet18().to(device)
        elif model_type == "wide_resnet_28_10":
            model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
        
        # Loading Dataset
        if data_type == "stl10":
            transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataset = STL10(root="./data", split="test",  download=True, transform=transform)
            test_dataset  = STL10(root="./data", split="train", download=True, transform=transform)
        elif data_type == "cifar100":
            transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataset = CIFAR100(root="./data", train=True,  transform=transform, download=True)
            test_dataset  = CIFAR100(root="./data", train=False, transform=transform, download=True)
        elif data_type == "cifar10":
            transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataset = CIFAR10(root="./data", train=True,  transform=transform, download=True)
            test_dataset  = CIFAR10(root="./data", train=False, transform=transform, download=True)
        
        n_samples = len(train_dataset)
        n_train   = int(n_samples * 0.8)
        n_val     = n_samples - n_train
        train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

        # Augmentation List
        augmentations = {
            # "Original",
            # "Mixup",
            # "Mixup-Original",
            # "Mixup-PCA",
            "Mixup-Original&PCA",
            
            # "Manifold-Mixup",
            # "PCA",
            # "FOMA",
        }

        for augment in augmentations:
            print(f"\n==> Training with {augment} ...")

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())
            score     = 0.0
            history   = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

            os.makedirs(f"./logs/{model_type}/{augment}",    exist_ok=True)
            os.makedirs(f"./history/{model_type}/{augment}", exist_ok=True)

            # TRAINING #
            for epoch in range(epochs):
                train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, augment, aug_ok=False, epochs=epoch)
                val_loss, val_acc     = val(model, val_loader, criterion, device, augment, aug_ok=False)

                if score <= val_acc:
                    print("Save model parameters...")
                    score = val_acc
                    model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"
                    torch.save(model.state_dict(), model_save_path)
                
                history["loss"].append(train_loss)
                history["accuracy"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                print(f"| {epoch+1} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f} |")

            with open(f"./history/{model_type}/{augment}/{data_type}_{epochs}_{i}.pickle", "wb") as f:
                pickle.dump(history, f)
            
            # TEST #
            model.load_state_dict(torch.load(model_save_path, weights_only=True))
            model.eval()
            test_loss, test_acc = test(model, test_loader, criterion, device, augment, aug_ok=False)
            print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")

            test_history = {"acc": test_acc, "loss": test_loss}
            with open(f"./history/{model_type}/{augment}/{data_type}_{epochs}_{i}_test.pickle", "wb") as f:
                pickle.dump(test_history, f)

def train(model, train_loader, criterion, optimizer, device, augment, aug_ok, epochs):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0

    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        if augment == "Original":  
            preds = model(images, labels, device, augment, aug_ok)
            loss  = criterion(preds, labels)

        elif augment == "Mixup":
            images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "Manifold-Mixup":
            preds, y_a, y_b, lam = model(images, labels, device, augment, mixup_hidden=True)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        
        elif augment == "Mixup-Original":
            if epochs < 200:
                images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
                preds = model(images, labels, device, augment, aug_ok)
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            else:
                preds = model(images, labels, device, augment, aug_ok)
                loss  = criterion(preds, labels)

        elif augment == "PCA":
            if epochs < 100:
                preds = model(images, labels, device, augment, aug_ok=False)
                loss  = criterion(preds, labels)
            else:
                preds = model(images, labels, device, augment, aug_ok=True)
                loss  = criterion(preds, labels)

        elif augment == "Mixup-PCA":
            if epochs < 200:
                images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
                preds = model(images, labels, device, augment, aug_ok)
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            else:
                preds = model(images, labels, device, augment, aug_ok=True)
                loss  = criterion(preds, labels)
        
        elif augment == "Mixup-Original&PCA":
            if epochs < 200:
                images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
                preds = model(images, labels, device, augment, aug_ok)
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            else:
                preds = model(images, labels, device, augment, aug_ok=True)
                loss  = criterion(preds, labels)

        elif augment == "FOMA":
            images, labels = foma_inputspace_per_class(images, labels, num_classes=10)
            preds          = model(images, labels, device, augment, aug_ok=False)
            loss           = - (labels * F.log_softmax(preds, dim=1)).sum(dim=1).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_score(labels.cpu(), preds.argmax(dim=-1).cpu())
        
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