import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.models.resnet import ResNet18, ResNet101
from src.models.wide_resnet import Wide_ResNet
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from sklearn.metrics import accuracy_score

# --- Calibration utilities ---
def compute_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE)
    confidences: shape (N,), model confidence for predicted class
    labels:      shape (N,), true labels
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    ece = 0.0
    N = len(confidences)
    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (confidences > lower) & (confidences <= upper)
        prop_in_bin = mask.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (labels[mask] == labels[mask]).mean() if False else None
            # Actually compute accuracy of predictions matching true labels
            # But here confidences correspond to predicted class confidence,
            # so we need predictions array separately. We'll compute ECE later with preds.
            pass
    # We'll implement combined function below.
    return ece


def reliability_diagram(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 15, savepath: str = "reliability_diagram.png"):  
    # Compute per-bin accuracy and confidence
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    accuracies = []
    avg_confidences = []
    bin_prop = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() > 0:
            accuracies.append((predictions[mask] == labels[mask]).mean())
            avg_confidences.append(confidences[mask].mean())
            bin_prop.append(mask.mean())
        else:
            accuracies.append(0)
            avg_confidences.append(0)
            bin_prop.append(0)
    # Plot
    plt.figure(figsize=(6,6))
    plt.plot([0,1], [0,1], linestyle='--', label='Perfect Calibration')
    plt.plot(bin_centers, accuracies, marker='o', label='Accuracy')
    plt.plot(bin_centers, avg_confidences, marker='x', label='Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def compute_ece_from_preds(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 50) -> float:
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


# --- Main testing with calibration ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--data_type", type=str, default="cifar100", choices=["stl10", "cifar100", "cifar10"])
    parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
    args = parser.parse_args()

    epochs = args.epochs
    data_type = args.data_type
    model_type = args.model_type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & loader
    if data_type == "stl10":
        num_classes, batch_size = 10, 64
        test_dataset = STL10(root="./data", split="train", download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                             ]))
    elif data_type == "cifar100":
        num_classes, batch_size = 100, 128
        test_dataset = CIFAR100(root="./data", train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                                ]))
    else:
        num_classes, batch_size = 10, 128
        test_dataset = CIFAR10(root="./data", train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                               ]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for augment in ["Default","Mixup", "Manifold-Mixup", "Local-FOMA"]:
        print(f"\n==> Test with {augment} ...")
        # Model
        if model_type == "resnet18": model = ResNet18().to(device)
        elif model_type == "resnet101": model = ResNet101().to(device)
        else: model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)

        model_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_0.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        all_confidences = []
        all_preds = []
        all_labels = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, targets, device, augment)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                probs = F.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, dim=1)

                all_confidences.append(conf.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        all_confidences = np.concatenate(all_confidences)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        avg_loss = total_loss / len(test_dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        ece = compute_ece_from_preds(all_confidences, all_preds, all_labels, n_bins=50)

        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, ECE: {ece:.4f}")
        reliability_diagram(
            all_confidences, all_preds, all_labels,
            n_bins=50,
            savepath=f"reliability_{augment}.png"
        )

if __name__ == "__main__":
    main()
