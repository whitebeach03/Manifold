import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from src.models.wide_resnet import Wide_ResNet
from src.models.resnet import ResNet18
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from src.utils import test
from tqdm import tqdm
import argparse
import numpy as np
import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.utils import *
from src.models.resnet import ResNet18, ResNet101
from src.models.wide_resnet import Wide_ResNet
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split, Subset
from src.methods.foma import foma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i",          type=int, default=0)
    parser.add_argument("--epochs",     type=int, default=250)
    parser.add_argument("--augment",    type=str, default="Default")
    parser.add_argument("--data_type",  type=str, default="cifar100",  choices=["stl10", "cifar100", "cifar10"])
    parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
    parser.add_argument("--k_foma",     type=int, default=0)
    args = parser.parse_args() 

    i          = args.i
    epochs     = args.epochs
    augment    = args.augment
    data_type  = args.data_type
    model_type = args.model_type
    k_foma     = args.k_foma
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_type == "cifar100":
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif data_type == "cifar10":
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std),
    ])
            
    # Loading Dataset
    if data_type == "stl10":
        num_classes = 10
        batch_size  = 64
        test_dataset = STL10(root="./data", split="train", download=True, transform=transform)
    elif data_type == "cifar100":
        epochs      = 250
        num_classes = 100
        batch_size  = 128
        test_dataset   = CIFAR100(root="./data", train=False, transform=transform, download=True)
    elif data_type == "cifar10":
        epochs      = 250
        num_classes = 10
        batch_size  = 128
        test_dataset   = CIFAR10(root="./data", train=False, transform=transform, download=True)

    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
    
    total_acc = 0
    total_loss = 0
    print(f"\n==> Test with {augment} ...")

    # Select Model
    if model_type == "resnet18":
        model = ResNet(PreActBlock, [2,2,2,2], num_classes).to(device)
    elif model_type == "resnet101":
        model = ResNet101().to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()

    if k_foma == 0:
        model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"
    else:    
        model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}_{k_foma}.pth"

    checkpoint = torch.load(model_save_path, weights_only=True)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    test_loss, test_acc = test(model, test_loader, criterion, device, augment)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc*100:.4f}")

class CIFAR100C(Dataset):
    def __init__(self, corruption_type, severity=1, root='./data/CIFAR-100-C', transform=None):
        self.data = np.load(f"{root}/{corruption_type}.npy")
        self.labels = np.load(f"{root}/labels.npy")
        assert 1 <= severity <= 5
        self.data = self.data[(severity - 1) * 10000 : severity * 10000]
        self.labels = self.labels[(severity - 1) * 10000 : severity * 10000]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = img.astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        return img, label
    
class CIFAR10C(Dataset):
    def __init__(self, corruption_type, severity=1, root='./data/CIFAR-10-C', transform=None):
        self.data = np.load(f"{root}/{corruption_type}.npy")
        self.labels = np.load(f"{root}/labels.npy")
        assert 1 <= severity <= 5
        self.data = self.data[(severity - 1) * 10000 : severity * 10000]
        self.labels = self.labels[(severity - 1) * 10000 : severity * 10000]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = img.astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        return img, label

if __name__ == "__main__":
    main()
