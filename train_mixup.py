import argparse
import numpy as np
import random
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
from tqdm import tqdm
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split, Subset

# FOMA関連のインポートは削除

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i",          type=int, default=0)
    parser.add_argument("--epochs",     type=int, default=250)
    parser.add_argument("--data_type",  type=str, default="cifar100",  choices=["stl10", "cifar100", "cifar10"])
    parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
    # k_foma 引数は削除
    args = parser.parse_args() 

    i          = args.i
    epochs     = args.epochs
    data_type  = args.data_type
    model_type = args.model_type
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    augment    = "Mixup" # augment名を変更
    
    set_seed(i)
    
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

    train_dataset = torch.utils.data.Subset(full_train_aug, train_indices)
    val_dataset   = torch.utils.data.Subset(full_train_plain, val_indices)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False)     
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
    
    print(f"\n==> Training with Mixup (Phase 2 Baseline) ...")

    # Select Model
    if model_type == "resnet18":
        model = ResNet18().to(device)
    elif model_type == "resnet101":
        model = ResNet101().to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
    
    start_epoch = 225
    train_epoch = 25
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = 0.1
    # スケジューラの引継ぎ設定（last_epochを指定）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=start_epoch-1)
    
    score     = 0.0
    history   = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    # 保存先ディレクトリ名を変更 (Mixup-FOMA -> Mixup)
    os.makedirs(f"./logs/{model_type}/Mixup",    exist_ok=True)
    os.makedirs(f"./history/{model_type}/Mixup", exist_ok=True)
    
    # Phase 1のモデルをロード
    mixup_save_path = f"./logs/wide_resnet_28_10/Mixup/{data_type}_{start_epoch}_{i}.pth"
    # model.load_state_dict(torch.load(mixup_save_path, weights_only=True))
    
    checkpoint = torch.load(mixup_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # 必要なら

    start_epoch = checkpoint['epoch'] + 1

    ### TRAINING ###
    for epoch in range(train_epoch):
        # train関数をMixup専用のものに変更
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, augment, num_classes, aug_ok=False)
        val_loss, val_acc     = val(model, val_loader, criterion, device, augment, aug_ok=False)
        scheduler.step()

        if score <= val_acc:
            print("Save model parameters...")
            score = val_acc
            # ファイル名も区別できるように変更
            model_save_path = f"./logs/{model_type}/Mixup/{data_type}_{epochs}_{i}.pth"
            torch.save(model.state_dict(), model_save_path)
        
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"| {start_epoch + epoch + 1} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f} |")

    with open(f"./history/{model_type}/Mixup/{data_type}_{epochs}_{i}.pickle", "wb") as f:
        pickle.dump(history, f)
    
    ### TEST ###
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_loss, test_acc = test(model, test_loader, criterion, device, augment=None, aug_ok=False)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.5f}")

    test_history = {"acc": test_acc, "loss": test_loss}
    with open(f"./history/{model_type}/Mixup/{data_type}_{epochs}_{i}_test.pickle", "wb") as f:
        pickle.dump(test_history, f)

def train(model, train_loader, criterion, optimizer, device, augment, num_classes, aug_ok):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0

    # enumerate でバッチを回す
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, leave=False)):
        images, labels = images.to(device), labels.to(device)
        labels_true = labels

        # === Phase 2: Plain Mixup (Baseline) ===
        # FOMAの処理（Feature Extraction, Memory Bank, Warm-up）を全て削除し、
        # シンプルなMixupのみを実行します。
        
        images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
        preds = model(images, labels, device, augment, aug_ok)
        loss  = mixup_criterion(criterion, preds, y_a, y_b, lam)

        # --- Optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Metrics ---
        train_loss += loss.item()
        y_pred = preds.argmax(dim=1)
        batch_acc = (y_pred == labels_true).float().mean().item()
        train_acc += batch_acc

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
            y_pred = preds.argmax(dim=1)
            batch_acc = (y_pred == labels).float().mean().item()
            val_acc += batch_acc

    val_loss /= len(val_loader)
    val_acc  /= len(val_loader)
    return val_loss, val_acc

def test(model, test_loader, criterion, device, augment, aug_ok):
    model.eval()
    test_loss = 0.0
    test_acc  = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            preds = model(images, labels, device, augment, aug_ok)
            loss  = criterion(preds, labels)
            test_loss += loss.item()
            y_pred = preds.argmax(dim=1)
            batch_acc = (y_pred == labels).float().mean().item()
            test_acc += batch_acc

    test_loss /= len(test_loader)
    test_acc  /= len(test_loader)
    return test_loss, test_acc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    main()