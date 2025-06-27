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
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split, Subset
from foma import foma
from batch_sampler import extract_wrn_features, FeatureKNNBatchSampler, HybridFOMABatchSampler

augmentations = [
    # "Default",
    # "Mixup",
    # "Manifold-Mixup",

    "Mixup-Curriculum"

    # "FOMA",
    # "FOMA_latent_random",

    # "FOMA_default",
    # "FOMA_knn_input",
    # "FOMA_hard",
    # "FOMA_curriculum"
    # "FOMA_samebatch"
    # "FOMA_knn"

    # "Mixup-Original",
    # "Mixup-PCA",
    # "Mixup-Original&PCA",
    # "PCA",
]

def main():
    for i in range(1):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs",     type=int, default=400)
        parser.add_argument("--data_type",  type=str, default="cifar100",          choices=["stl10", "cifar100", "cifar10"])
        parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "wide_resnet_28_10"])
        args = parser.parse_args() 

        epochs     = args.epochs
        data_type  = args.data_type
        model_type = args.model_type
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
        
        default_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
                
        # Loading Dataset
        if data_type == "stl10":
            train_dataset = STL10(root="./data", split="test",  download=True, transform=default_transform)
            test_dataset  = STL10(root="./data", split="train", download=True, transform=transform)
        elif data_type == "cifar100":
            full_train_aug   = CIFAR100(root="./data", train=True,  transform=default_transform, download=True)
            full_train_plain = CIFAR100(root="./data", train=True,  transform=transform,         download=True)
            test_dataset     = CIFAR100(root="./data", train=False, transform=transform,         download=True)
        elif data_type == "cifar10":
            full_train_aug   = CIFAR10(root="./data", train=True,  transform=default_transform, download=True)
            full_train_plain = CIFAR10(root="./data", train=True,  transform=transform,         download=True)
            test_dataset     = CIFAR10(root="./data", train=False, transform=transform,         download=True)
        
        n_samples = len(full_train_aug)
        n_train   = int(n_samples * 0.8)
        n_val     = n_samples - n_train
        train_indices, val_indices = random_split(range(n_samples), [n_train, n_val])

        train_dataset = torch.utils.data.Subset(full_train_aug, train_indices)
        val_dataset   = torch.utils.data.Subset(full_train_plain, val_indices)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False)     
        test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
        
        for augment in augmentations:
            print(f"\n==> Training with {augment} ...")

            # Select Model
            if model_type == "resnet18":
                model = ResNet18().to(device)
            elif model_type == "wide_resnet_28_10":
                model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
                
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())
            score     = 0.0
            history   = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

            os.makedirs(f"./logs/{model_type}/{augment}",    exist_ok=True)
            os.makedirs(f"./history/{model_type}/{augment}", exist_ok=True)

            ### TRAINING ###
            for epoch in range(epochs):
                train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, augment, num_classes, aug_ok=False, epochs=epoch)
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
            
            ### TEST ###
            model.load_state_dict(torch.load(model_save_path, weights_only=True))
            test_loss, test_acc = test(model, test_loader, criterion, device, augment, aug_ok=False)
            print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")

            test_history = {"acc": test_acc, "loss": test_loss}
            with open(f"./history/{model_type}/{augment}/{data_type}_{epochs}_{i}_test.pickle", "wb") as f:
                pickle.dump(test_history, f)

if __name__ == "__main__":
    main()