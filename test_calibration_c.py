import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from typing import Dict
from test_acc import CIFAR10C, CIFAR100C  # 事前インポート済み
from src.models.resnet import ResNet18, ResNet101
from src.models.wide_resnet import Wide_ResNet
import matplotlib.pyplot as plt



def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corruption_list = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if args.data_type == "cifar10":
        dataset_class = CIFAR10C
        num_classes, epochs = 10, 250
    elif args.data_type == "cifar100":
        dataset_class = CIFAR100C
        num_classes, epochs = 100, 400
    else:
        raise ValueError("Invalid data_type")

    batch_size = 512
    model = get_model(args.model_type, num_classes=num_classes).to(device)
    model_path = f"./logs/{args.model_type}/{args.augment}/{args.data_type}_{epochs}_{args.iter}.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))

    ece_list = []
    nll_list = []
    brier_list = []

    for corruption in corruption_list:
        print(f"\n==> Evaluating corruption: {corruption}")
        dataset = dataset_class(corruption_type=corruption, severity=args.severity, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        ece, nll, brier = evaluate(model, dataloader, device, args.augment, num_classes)
        print(f"  ECE: {ece:.4f}, NLL: {nll:.4f}, Brier: {brier:.4f}")

        ece_list.append(ece)
        nll_list.append(nll)
        brier_list.append(brier)

    print("\n=== Final Summary ===")
    print(f"Average ECE   : {np.mean(ece_list):.4f}")
    print(f"Average Brier : {np.mean(brier_list):.4f}")
    print(f"Average NLL   : {np.mean(nll_list):.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--data_type", type=str, required=True,
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--model_type", type=str, default="wide_resnet_28_10",
                        choices=["resnet18", "resnet101", "wide_resnet_28_10"])
    parser.add_argument("--augment", type=str, default="Default")
    parser.add_argument("--severity", type=int, default=5)
    # parser.add_argument("--epochs", type=int, default=400)
    return parser.parse_args()


def get_model(model_type: str, num_classes: int):
    if model_type == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif model_type == "resnet101":
        return ResNet101(num_classes=num_classes)
    else:
        return Wide_ResNet(28, 10, 0.3, num_classes)


def evaluate(model, dataloader, device, augment, num_classes: int):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_confidences = []
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, targets, device, augment)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)

            all_probs.append(probs.cpu())
            all_confidences.append(conf.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    all_confidences = np.concatenate(all_confidences)
    all_preds = np.concatenate(all_preds)
    all_labels_np = np.concatenate(all_labels)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels_tensor = torch.tensor(all_labels_np, dtype=torch.long)

    ece = compute_ece_from_preds(all_confidences, all_preds, all_labels_np)
    log_probs = torch.log(all_probs + 1e-12)
    nll = F.nll_loss(log_probs, all_labels_tensor, reduction='mean').item()
    one_hot = F.one_hot(all_labels_tensor, num_classes=num_classes).float()
    brier = torch.mean(torch.sum((all_probs - one_hot) ** 2, dim=1)).item()

    return ece, nll, brier


def compute_ece_from_preds(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidences)
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        prop_in_bin = mask.sum() / N
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[mask] == labels[mask]).mean()
            avg_confidence_in_bin = confidences[mask].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

if __name__ == "__main__":
    main()
