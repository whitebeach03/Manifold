import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.models.wide_resnet import Wide_ResNet
from src.models.resnet import ResNet18, ResNet101
from torchvision.datasets import STL10, CIFAR10, CIFAR100


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

    # Data Settings
    if data_type == "cifar100":
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        num_classes = 100
        test_dataset = CIFAR100(root="./data", train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]), download=True)
    elif data_type == "cifar10":
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        num_classes = 10
        test_dataset = CIFAR10(root="./data", train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]), download=True)
    elif data_type == "stl10":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        num_classes = 10
        test_dataset = STL10(root="./data", split="test", transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]), download=True)

    # Model Setup
    print(f"Loading {model_type} trained with {augment}...")
    if model_type == "resnet18":
        model = ResNet18().to(device)
    elif model_type == "resnet101":
        model = ResNet101().to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)

    # Load Weights
    if k_foma == 0:
        model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"
    else:    
        model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}_{k_foma}.pth"
    
    if not os.path.exists(model_save_path):
        print(f"Error: Model checkpoint not found at {model_save_path}")
        return

    checkpoint = torch.load(model_save_path, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # --- 実験設定 ---
    # Lambdaを 0.1 から 1.0 まで
    lambda_values = np.linspace(0.1, 1.0, 10)
    
    # Class-wise SVD 評価実行
    accuracies = evaluate_class_wise_svd(model, test_dataset, device, lambda_values, num_classes)

    # 結果のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, accuracies, marker='o', label=f'{augment} (Class-wise SVD)')
    plt.title(f'Effect of Class-wise Manifold Projection on {data_type}')
    plt.xlabel(r'Lambda $\lambda$ (Fraction of Singular Values Kept)')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    # 画像を保存
    save_img_path = f"./logs/{model_type}/{augment}/class_wise_svd_{data_type}.png"
    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    plt.savefig(save_img_path)
    print(f"\nResult plot saved to {save_img_path}")

def project_data_to_manifold(images, lambda_val):
    """
    データ群(クラス単位など)に対してSVDを行い、特異値を操作して再構成する
    """
    if lambda_val >= 1.0:
        return images
    
    device = images.device
    B, C, H, W = images.shape
    flat_images = images.view(B, -1) # (samples, features)
    
    # SVD実行 (CPUで行ったほうがメモリ安全な場合が多いですが、サイズ次第でGPUも可)
    # ここではデータ量が多い可能性を考慮してCPU計算を推奨しますが、
    # CIFARクラス単位(1000枚)ならGPUでも余裕です。
    
    # U: (B, B), S: (min(B, D)), Vh: (D, D)
    # full_matrices=False なので Sのサイズは小さい方に合わせられる
    U, S, Vh = torch.linalg.svd(flat_images, full_matrices=False)
    
    # 上位 lambda_val 割合を残す
    num_keep = int(len(S) * lambda_val)
    if num_keep < 1: num_keep = 1
    
    S_mod = S.clone()
    S_mod[num_keep:] = 0
    
    # 再構成
    reconstructed_flat = U @ (torch.diag(S_mod) @ Vh)
    
    return reconstructed_flat.view(B, C, H, W)

def evaluate_class_wise_svd(model, dataset, device, lambda_values, num_classes):
    """
    クラスごとにデータをまとめてSVD射影してから評価する関数
    """
    accuracies = []
    model.eval()
    
    # データセットのターゲット（正解ラベル）を取得
    # STL10などは .labels, CIFARは .targets という属性名の場合があるため対応
    if hasattr(dataset, 'targets'):
        all_targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        all_targets = np.array(dataset.labels)
    else:
        raise ValueError("Dataset does not have .targets or .labels attribute")

    print("\nStarting Class-wise SVD Manifold Experiment...")

    # 事前に全データをメモリにロードしてしまう（CIFAR/STL程度なら可）
    # transformを適用するためにDataLoaderを一度通すのが安全ですが、
    # ここでは簡易的に全データを取得するLoaderを作ります
    full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)
    all_images_tensor, all_labels_tensor = next(iter(full_loader))
    
    # lambdaごとにループ
    for lam in lambda_values:
        total_correct = 0
        total_samples = 0
        
        # クラスごとに処理
        # tqdmの表示を工夫
        pbar = tqdm(range(num_classes), desc=f"Lambda: {lam:.2f}", leave=False)
        
        for class_idx in pbar:
            # そのクラスに該当するインデックスを取得
            indices = np.where(all_targets == class_idx)[0]
            
            # 該当データを抽出 (GPUに送って計算)
            class_images = all_images_tensor[indices].to(device)
            class_labels = all_labels_tensor[indices].to(device)
            
            # === ここが重要: クラスごとの多様体への射影 ===
            # クラス内の全データを使ってSVD基底を作る
            projected_images = project_data_to_manifold(class_images, lam)
            
            # 推論 (メモリ溢れを防ぐため、さらにミニバッチに分けて推論する)
            # 1000枚程度なら一気に入ることも多いですが、安全策で分割
            mini_batch_size = 128
            num_samples = projected_images.size(0)
            
            with torch.no_grad():
                for i in range(0, num_samples, mini_batch_size):
                    end = min(i + mini_batch_size, num_samples)
                    batch_img = projected_images[i:end]
                    batch_lbl = class_labels[i:end]
                    
                    outputs = model(batch_img, batch_lbl, device, augment=None)
                    _, predicted = outputs.max(1)
                    
                    total_correct += predicted.eq(batch_lbl).sum().item()
                    total_samples += batch_lbl.size(0)

        acc = 100. * total_correct / total_samples
        accuracies.append(acc)
        print(f"Lambda: {lam:.2f} | Accuracy: {acc:.2f}%")
        
    return accuracies


if __name__ == "__main__":
    main()