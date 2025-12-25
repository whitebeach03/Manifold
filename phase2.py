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
from tqdm import tqdm
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split, Subset

from src.utils import *
from src.models.resnet import ResNet18, ResNet101
from src.models.wide_resnet import Wide_ResNet
from src.methods.cc_foma import cc_foma
from src.memory_bank import FeatureMemoryBank
from src.methods.foma import local_foma, local_foma_fast, local_foma_fast_with_memory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i",            type=int,   default=0)
    parser.add_argument("--epochs",       type=int,   default=250,                 help="Total epochs (Phase1 + Phase2)")
    parser.add_argument("--data_type",    type=str,   default="cifar100",          choices=["stl10", "cifar100", "cifar10"])
    parser.add_argument("--model_type",   type=str,   default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
    parser.add_argument("--method",       type=str,   default="ES-Mixup",          choices=["ES-Mixup", "Mixup", "Mixup-FOMA", "Mixup-FOMA2"], help="Phase 2 method: ES-Mixup (Clean), Mixup (Continue), or Mixup-FOMA")
    parser.add_argument("--phase1_ratio", type=float, default=0.9,                 help="Ratio of Phase 1 epochs")
    parser.add_argument("--k_foma",       type=int,   default=8,                   help="k-neighbors for FOMA")
    
    args = parser.parse_args() 

    i            = args.i
    epochs       = args.epochs
    data_type    = args.data_type
    model_type   = args.model_type
    method       = args.method
    phase1_ratio = args.phase1_ratio
    k_foma       = args.k_foma
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        full_train_aug = STL10(root="./data", split="test",  download=True, transform=default_transform)
        test_dataset   = STL10(root="./data", split="train", download=True, transform=transform)
        full_train_plain = STL10(root="./data", split="test", download=True, transform=transform) 
    elif data_type == "cifar100":
        full_train_aug   = CIFAR100(root="./data", train=True,  transform=default_transform, download=True)
        full_train_plain = CIFAR100(root="./data", train=True,  transform=transform,         download=True)
        test_dataset     = CIFAR100(root="./data", train=False, transform=transform,         download=True)
    elif data_type == "cifar10":
        full_train_aug   = CIFAR10(root="./data", train=True,  transform=default_transform, download=True)
        full_train_plain = CIFAR10(root="./data", train=True,  transform=transform,         download=True)
        test_dataset     = CIFAR10(root="./data", train=False, transform=transform,         download=True)

    # Split Data
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
    
    # DataLoader 
    g = torch.Generator()
    g.manual_seed(i)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn, generator=g)     
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    
    print(f"\n==> Phase 2 Training Method: {method}")

    # Select Model
    if model_type == "resnet18":
        model = ResNet18().to(device)
    elif model_type == "resnet101":
        model = ResNet101().to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
    
    start_epoch_phase1 = int(epochs * phase1_ratio)
    train_epoch_phase2 = epochs - start_epoch_phase1

    print(f"Total Epochs: {epochs}")
    print(f"Phase 1 Epochs (Load point): {start_epoch_phase1}")
    print(f"Phase 2 Epochs (Training): {train_epoch_phase2}")
        
    mixup_save_path = f"./logs/{model_type}/Mixup/{data_type}_{start_epoch_phase1}_{i}.pth"
    print(f"Loading Phase 1 model from {mixup_save_path} ...")
    
    checkpoint = torch.load(mixup_save_path, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        current_start_epoch = checkpoint['epoch'] + 1
    else:
        model.load_state_dict(checkpoint)
        current_start_epoch = start_epoch_phase1 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = 0.1
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=current_start_epoch-1)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Resume Learning Rate: {current_lr}")
    
    score     = 0.0
    history   = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    # Logging Directories
    save_dir_name = method
    os.makedirs(f"./logs/{model_type}/{save_dir_name}",    exist_ok=True)
    os.makedirs(f"./history/{model_type}/{save_dir_name}", exist_ok=True)
    
    # ==========================================
    # FOMA Setup (Memory Bank)
    # ==========================================
    memory_bank = None
    if method == "Mixup-FOMA" or method == "Mixup-FOMA2":
        print("==> Initializing Feature Memory Bank...")
        feature_dim = model.linear.in_features
        memory_bank = FeatureMemoryBank(feature_dim=feature_dim, memory_size=5000, num_classes=num_classes)
        
        print("==> Filling Memory Bank (Warm-up)...")
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Memory Bank Init", leave=False):
                images, labels = images.to(device), labels.to(device)
                features = model.extract_features(images)
                memory_bank.update(features, labels)
        print("==> Memory Bank is ready!")

    for epoch in range(train_epoch_phase2):
        train_loss, train_acc = train_phase2(
            model=model, 
            train_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            method=method, 
            num_classes=num_classes, 
            relative_epoch=epoch,
            total_epochs_phase2=train_epoch_phase2,
            k_foma=k_foma,
            memory_bank=memory_bank
        )
        
        val_loss, val_acc = val(model, val_loader, criterion, device, augment=method)
        scheduler.step()

        if score <= val_acc:
            print("Save model parameters...")
            score = val_acc
            if method == "Mixup-FOMA" or method == "Mixup-FOMA2":
                model_save_path = f"./logs/{model_type}/{save_dir_name}/{data_type}_{epochs}_{i}_{k_foma}.pth"
            else:
                model_save_path = f"./logs/{model_type}/{save_dir_name}/{data_type}_{epochs}_{i}.pth"
            
            torch.save(model.state_dict(), model_save_path)
        
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        
        print(f"| {current_start_epoch + epoch} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f} |")

    with open(f"./history/{model_type}/{save_dir_name}/{data_type}_{epochs}_{i}.pickle", "wb") as f:
        pickle.dump(history, f)

    # ==========================================
    # Save Phase 2 History & Merge
    # ==========================================
    
    phase2_history_path = f"./history/{model_type}/{save_dir_name}/{data_type}_{epochs}_{i}.pickle"

    with open(phase2_history_path, "wb") as f:
        pickle.dump(history, f)

    # Phase 1との統合処理
    print("\n==> Merging Phase 1 and Phase 2 history...")
    
    phase1_history_path = f"./history/{model_type}/Mixup/{data_type}_{start_epoch_phase1}_{i}.pickle"
    
    if os.path.exists(phase1_history_path):
        try:
            with open(phase1_history_path, "rb") as f:
                h1 = pickle.load(f)
            
            h2 = history
            
            merged_history = {}
            keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
            
            for key in keys:
                if key in h1 and key in h2:
                    merged_history[key] = h1[key] + h2[key]
            
            with open(phase2_history_path, "wb") as f:
                pickle.dump(merged_history, f)
                
            print(f"Successfully merged! Full history saved to:")
            print(f" -> {phase2_history_path}")
            
        except Exception as e:
            print(f"Warning: Failed to merge histories. Error: {e}")
    else:
        print(f"Warning: Phase 1 history file not found at {phase1_history_path}.")
    
    ### TEST ###
    print(f"Loading best model from {model_save_path} for testing...")
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_loss, test_acc = test(model, test_loader, criterion, device, augment=method)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.5f}")

    test_history = {"acc": test_acc, "loss": test_loss}
    with open(f"./history/{model_type}/{save_dir_name}/{data_type}_{epochs}_{i}_test.pickle", "wb") as f:
        pickle.dump(test_history, f)


def train_phase2(model, train_loader, criterion, optimizer, device, method, num_classes, relative_epoch, total_epochs_phase2, k_foma=32, memory_bank=None):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0

    w_clean = 0.5
    w_foma = 0.5

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, leave=False)):
        images, labels = images.to(device), labels.to(device)
        labels_true = labels
        loss = 0.0
        preds_for_acc = None
        
        if method == "Mixup-FOMA":
            with torch.no_grad():
                features_raw = model.extract_features(images)
            if memory_bank is not None:
                memory_bank.update(features_raw, labels)
            
            # 1. Clean Loss
            features_clean = model.extract_features(images)
            preds_clean = model.linear(features_clean)
            loss_clean = criterion(preds_clean, labels)
            preds_for_acc = preds_clean

            # 2. FOMA Loss 
            z_aug = cc_foma(features_clean, labels, memory_bank, k=k_foma, alpha=1.0, rho=0.9)
            preds_aug = model.linear(z_aug)
            loss_foma = criterion(preds_aug, labels)

            loss = w_clean*loss_clean + w_foma*loss_foma

        elif method == "Mixup-FOMA2":
            with torch.no_grad():
                features_raw = model.extract_features(images)
            if memory_bank is not None:
                memory_bank.update(features_raw, labels)
            # 1. Clean Loss

            features_clean = model.extract_features(images)
            preds_clean = model.linear(features_clean)
            loss_clean = criterion(preds_clean, labels) # 通常のHard Label Loss
            preds_for_acc = preds_clean

            # 2. Unrestricted FOMA (Soft Labels)

            z_aug, y_aug_soft = local_foma_fast_with_memory(
                features_clean, 
                labels, 
                memory_bank=memory_bank,
                k=k_foma, 
                alpha=1.0, 
                rho=0.9, 
                num_classes=num_classes
            )
            
            preds_aug = model.linear(z_aug)
            
            # ソフトラベル用の損失関数を使用
            loss_foma = soft_cross_entropy(preds_aug, y_aug_soft)

            loss = w_clean * loss_clean + w_foma * loss_foma

        elif method == "Mixup":
            images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            preds_mix = model(images, labels, device, augment="Mixup")
            loss = mixup_criterion(criterion, preds_mix, y_a, y_b, lam)
            preds_for_acc = preds_mix

        elif method == "ES-Mixup":
            preds = model(images, labels, device, augment=None)
            loss  = criterion(preds, labels)
            preds_for_acc = preds

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if preds_for_acc is not None:
            y_pred = preds_for_acc.argmax(dim=1)
            batch_acc = (y_pred == labels_true).float().mean().item()
            train_acc += batch_acc

    train_loss /= len(train_loader)
    train_acc  /= len(train_loader)
    
    return train_loss, train_acc

def soft_cross_entropy(pred, soft_targets):
    """
    ソフトラベル対応のクロスエントロピー誤差
    Args:
        pred: モデルの出力 (Logits)
        soft_targets: ソフトラベル (確率分布, Sum=1)
    """
    logsoftmax = F.log_softmax(pred, dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax, dim=1))

if __name__ == "__main__":
    main()