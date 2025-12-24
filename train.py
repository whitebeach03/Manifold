import argparse
import numpy as np
import random
import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Subset

from src.utils import train, val, test, seed_everything, worker_init_fn
from src.models.resnet import ResNet18, ResNet101
from src.models.wide_resnet import Wide_ResNet

def main():
    parser = argparse.ArgumentParser(description="General Training (Baselines)")
    parser.add_argument("--i",          type=int, default=0, help="Seed index")
    parser.add_argument("--epochs",     type=int, default=250, help="Total training epochs")
    parser.add_argument("--augment",    type=str, default="Default", choices=["Default", "Mixup", "CutMix", "Manifold-Mixup", "ResizeMix", "SaliencyMix", "SK-Mixup"], help="Augmentation method")
    parser.add_argument("--data_type",  type=str, default="cifar100",  choices=["stl10", "cifar100", "cifar10"])
    parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
    args = parser.parse_args() 

    i          = args.i
    epochs     = args.epochs
    augment    = args.augment
    data_type  = args.data_type
    model_type = args.model_type
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seed_everything(i)
    g = torch.Generator()
    g.manual_seed(i)

    if data_type == "stl10":
        num_classes = 10
        batch_size  = 64
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif data_type == "cifar100":
        num_classes = 100
        batch_size  = 128
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif data_type == "cifar10":
        num_classes = 10
        batch_size  = 128
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    
    default_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std),
    ])
            
    # Loading Dataset
    if data_type == "stl10":
        full_train_aug = STL10(root="./data", split="train",  download=True, transform=default_transform)
        full_train_plain = STL10(root="./data", split="train", download=True, transform=transform)
        test_dataset   = STL10(root="./data", split="test",  download=True, transform=transform)
    elif data_type == "cifar100":
        full_train_aug   = CIFAR100(root="./data", train=True,  transform=default_transform, download=True)
        full_train_plain = CIFAR100(root="./data", train=True,  transform=transform,         download=True)
        test_dataset     = CIFAR100(root="./data", train=False, transform=transform,         download=True)
    elif data_type == "cifar10":
        full_train_aug   = CIFAR10(root="./data", train=True,  transform=default_transform, download=True)
        full_train_plain = CIFAR10(root="./data", train=True,  transform=transform,         download=True)
        test_dataset     = CIFAR10(root="./data", train=False, transform=transform,         download=True)

    # Split Indices
    n_samples = len(full_train_aug)
    n_train   = int(n_samples * 0.8)
    os.makedirs("./data_split", exist_ok=True)
    index_file = f"./data_split/split_indices_{data_type}_{i}.pkl"

    if os.path.exists(index_file):
        print(f"Loading split indices from {index_file}")
        with open(index_file, "rb") as f:
            train_indices, val_indices = pickle.load(f)
    else:
        print("Generating new split indices...")
        indices = list(range(n_samples))
        random.shuffle(indices) 
        train_indices, val_indices = indices[:n_train], indices[n_train:]
        with open(index_file, "wb") as f:
            pickle.dump((train_indices, val_indices), f)

    train_dataset = Subset(full_train_aug, train_indices)
    val_dataset   = Subset(full_train_plain, val_indices)
    
    # DataLoaders with Reproducibility
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn, generator=g)     
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    
    print(f"\n==> General Training with {augment} ...")

    # Select Model
    if model_type == "resnet18":
        model = ResNet18().to(device)
    elif model_type == "resnet101":
        model = ResNet101().to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    score     = 0.0
    history   = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    save_dir = f"./logs/{model_type}/{augment}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"./history/{model_type}/{augment}", exist_ok=True)

    ### TRAINING ###
    for epoch in range(epochs):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, device, 
            augment, num_classes, epochs=epoch
        )
        val_loss, val_acc = val(model, val_loader, criterion, device, augment="Default")
        scheduler.step()

        if score <= val_acc:
            print("Save model parameters...")
            score = val_acc
            model_save_path = f"{save_dir}/{data_type}_{epochs}_{i}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc
            }, model_save_path)
        
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"| {epoch+1}/{epochs} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f} |")

    with open(f"./history/{model_type}/{augment}/{data_type}_{epochs}_{i}.pickle", "wb") as f:
        pickle.dump(history, f)
    
    ### TEST ###
    print(f"Loading Best model from {model_save_path} ...")
    checkpoint = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = test(model, test_loader, criterion, device, augment=None)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.5f}")

    test_history = {"acc": test_acc, "loss": test_loss}
    with open(f"./history/{model_type}/{augment}/{data_type}_{epochs}_{i}_test.pickle", "wb") as f:
        pickle.dump(test_history, f)

if __name__ == "__main__":
    main()