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

augmentations = [
    # "Default",
    # "Mixup",
    "Mixup-FOMA2",
    # "Mixup-FOMA-scaleup",_
    # "Manifold-Mixup",
    # "Local-FOMA",
    # "FOMA-Mixup",
    # "RegMixup",
    # "FOMA-scaleup"
]

corruption_types = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]
severity = 5
def main():
    for i in range(2, 3):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs",     type=int, default=400)
        parser.add_argument("--data_type",  type=str, default="cifar100",  choices=["stl10", "cifar100", "cifar10"])
        parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
        args = parser.parse_args() 

        epochs     = args.epochs
        data_type  = args.data_type
        model_type = args.model_type
        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
                
        # Loading Dataset
        if data_type == "stl10":
            num_classes = 10
            batch_size  = 64
            test_dataset = STL10(root="./data", split="train", download=True, transform=transform)
        elif data_type == "cifar100":
            epochs      = 400
            num_classes = 100
            batch_size  = 128
            test_dataset   = CIFAR100(root="./data", train=False, transform=transform, download=True)
        elif data_type == "cifar10":
            epochs      = 250
            num_classes = 10
            batch_size  = 128
            test_dataset   = CIFAR10(root="./data", train=False, transform=transform, download=True)

        test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
        
        for augment in augmentations:
            total_acc = 0
            total_loss = 0
            print(f"\n==> Test with {augment} ...")

            # Select Model
            if model_type == "resnet18":
                model = ResNet18().to(device)
            elif model_type == "resnet101":
                model = ResNet101().to(device)
            elif model_type == "wide_resnet_28_10":
                model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
            
            criterion = nn.CrossEntropyLoss()

            model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"
            model.load_state_dict(torch.load(model_save_path, weights_only=True))
            test_loss, test_acc = test(model, test_loader, criterion, device, augment, aug_ok=False)
            print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")

            for corruption in corruption_types:
                if data_type == "cifar100":
                    test_dataset_C = CIFAR100C(corruption_type=corruption, severity=severity, transform=transform)
                elif data_type == "cifar10":
                    test_dataset_C = CIFAR10C(corruption_type=corruption, severity=severity, transform=transform)
                test_loader_C  = torch.utils.data.DataLoader(test_dataset_C, batch_size=512, shuffle=False)
                test_loss_C, test_acc_C = test(model, test_loader_C, criterion, device, augment, aug_ok=False)
                print(f"  [{corruption}] Loss: {test_loss_C:.3f}, Accuracy: {test_acc_C:.2f}%")
                total_acc += test_acc_C
                total_loss += test_loss_C

            avg_acc = total_acc / len(corruption_types)
            avg_loss = total_loss / len(corruption_types)

            print(f"\nLoss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}%")


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
