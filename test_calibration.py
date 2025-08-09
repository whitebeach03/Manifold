import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from src.models.resnet import ResNet18, ResNet101
from src.models.wide_resnet import Wide_ResNet
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from sklearn.metrics import accuracy_score
from matplotlib.colors import LinearSegmentedColormap

n_iteration = 3

augmentations = [
    # "Default",
    "Mixup",
    # "Manifold-Mixup",
    # "CutMix",
    # "Mixup-FOMA2",
    # "Local-FOMA",
    # "Mixup-FOMA-scaleup"
]

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

    n_bins = 20

    # Dataset & loader
    if data_type == "stl10":
        num_classes, batch_size = 10, 64
        test_dataset = STL10(root="./data", split="train", download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                              ]))
    elif data_type == "cifar100":
        num_classes, batch_size, epochs = 100, 128, 400
        test_dataset = CIFAR100(root="./data", train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                 ]))
    else:
        num_classes, batch_size, epochs = 10, 128, 250
        test_dataset = CIFAR10(root="./data", train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                ]))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for augment in augmentations:
        print(f"\n==> Test with {augment} ...")

        # Model
        if model_type == "resnet18":
            model = ResNet18().to(device)
        elif model_type == "resnet101":
            model = ResNet101().to(device)
        else:
            model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)

        model_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{n_iteration}.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        criterion = nn.CrossEntropyLoss()

        all_confidences = []
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
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

        # Convert to full tensors/arrays
        all_confidences = np.concatenate(all_confidences)
        all_preds = np.concatenate(all_preds)
        all_labels_np = np.concatenate(all_labels)
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.tensor(all_labels_np)

        # ECE
        ece = evaluate_calibration(
            confidences=all_confidences,
            predictions=all_preds,
            labels=all_labels_np,
            n_bins=n_bins,
            save_path=f"./ECE/{data_type}/{augment}_{n_iteration}.png"
        )

        # NLL
        log_probs = torch.log(all_probs + 1e-12)
        nll = F.nll_loss(log_probs, all_labels, reduction='mean').item()

        # Brier Score
        one_hot = F.one_hot(all_labels, num_classes=num_classes).float()
        brier = torch.mean(torch.sum((all_probs - one_hot) ** 2, dim=1)).item()

        print(f"ECE = {ece:.4f}, Brier Score = {brier:.4f}, NLL = {nll:.4f}")


    #     avg_loss = total_loss / len(test_dataset)
    #     accuracy = accuracy_score(all_labels, all_preds)
    #     ece = compute_ece_from_preds(all_confidences, all_preds, all_labels, n_bins=n_bins)

    #     print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, ECE: {ece:.4f}")
    #     classwise_ece = compute_classwise_ece(all_confidences, all_preds, all_labels, n_bins=n_bins)
    #     all_classwise_ece[augment] = classwise_ece
    #     reliability_diagram(
    #         all_confidences, all_preds, all_labels,
    #         n_bins=n_bins,
    #         savepath=f"./ECE/{data_type}/{augment}.png"
    #     )
        
    #     classwise_ece = compute_classwise_ece(all_confidences, all_preds, all_labels, n_bins=n_bins)
    #     # print(">> Class-wise ECE:")
    #     # for cls, ece_val in sorted(classwise_ece.items()):
    #     #     print(f"  Class {cls:3d}: ECE = {ece_val:.4f}")
    
    # methods = list(all_classwise_ece.keys())
    # num_classes = max(len(v) for v in all_classwise_ece.values())
    # # 行列データ作成（methods 行 × classes 列）
    # data = np.array([
    #     [ all_classwise_ece[m].get(c, 0.0) for c in range(num_classes) ]
    #     for m in methods
    # ])

    # plt.figure(figsize=(12, 3))
    # plt.imshow(data, aspect='auto', cmap='viridis')
    # plt.colorbar(label='ECE')
    # plt.yticks(np.arange(len(methods)), methods)
    # plt.xlabel('Class Label')
    # plt.title('Class-wise ECE Heatmap')
    # plt.tight_layout()
    # plt.savefig(f'./ECE/{data_type}/classwise_heatmap.png')

    # ece_values_by_method = [
    #     [all_classwise_ece[method].get(cls, np.nan) for cls in range(num_classes)]
    #     for method in methods
    # ]
    
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(ece_values_by_method, labels=methods, showfliers=True)
    # plt.ylabel("ECE")
    # plt.title("Class-wise ECE Distribution by Method (Boxplot)")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"./ECE/{data_type}/classwise_boxplot.png")


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
    # plt.plot(bin_centers, avg_confidences, marker='x', label='Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath)

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

def compute_classwise_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 50
) -> Dict[int, float]:
    """
    各クラスごとに ECE を計算して辞書で返す
    """
    class_ece: Dict[int, float] = {}
    for cls in np.unique(labels):
        mask = (labels == cls)
        confs_cls = confidences[mask]
        preds_cls = predictions[mask]
        labels_cls = labels[mask]
        # サンプル数が少ない場合はスキップも可
        if len(confs_cls) == 0:
            continue
        ece_cls = compute_ece_from_preds(confs_cls, preds_cls, labels_cls, n_bins)
        class_ece[int(cls)] = ece_cls
    return class_ece

def evaluate_calibration(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    save_path: str = "reliability_diagram.png",
    show: bool = False,
) -> float:
    """
    ECEを計算し、Reliability Diagramを描画・保存する関数

    Args:
        confidences: shape (N,), 各予測の予測クラスの信頼度
        predictions: shape (N,), モデルの予測ラベル
        labels:      shape (N,), 正解ラベル
        n_bins: ビンの数（default: 15）
        save_path: 図の保存パス
        show: plt.show() するかどうか

    Returns:
        ece: Expected Calibration Error のスカラー値
    """
    assert len(confidences) == len(predictions) == len(labels), "入力配列の長さが不一致"

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    accuracies = []
    avg_confidences = []
    bin_counts = []
    ece = 0.0
    N = len(confidences)

    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        bin_size = np.sum(mask)
        if bin_size > 0:
            avg_conf = confidences[mask].mean()
            acc = (predictions[mask] == labels[mask]).mean()
            weight = bin_size / N
            ece += np.abs(avg_conf - acc) * weight

            accuracies.append(acc)
            avg_confidences.append(avg_conf)
            bin_counts.append(bin_size)
        else:
            accuracies.append(0.0)
            avg_confidences.append(0.0)
            bin_counts.append(0)

    # 可視化
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    plt.plot(bin_centers, accuracies, marker='o', label='Accuracy', color='darkorange')
    # plt.bar(bin_centers, [c / max(bin_counts) for c in bin_counts], width=1/n_bins*0.8, alpha=0.2, label='Sample Ratio')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

    return ece

if __name__ == "__main__":
    main()
