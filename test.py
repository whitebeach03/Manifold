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

# class IndexedDataset(Dataset):
#     def __init__(self, base_dataset):
#         self.dataset = base_dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img, label = self.dataset[idx]
#         return img, label, idx

# def test_and_collect_errors(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     errors = []   # 誤分類リスト。sample_idx, true, pred を格納

#     with torch.no_grad():
#         for inputs, targets, indices in tqdm(loader, leave=False):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs, targets, device=device, augment="Mixup")
#             total_loss += criterion(outputs, targets).item() * inputs.size(0)
#             _, preds = torch.max(outputs, dim=1)

#             total_correct += (preds == targets).sum().item()
#             total_samples += targets.size(0)

#             # 誤分類をチェック
#             mismatch = preds != targets
#             for i in torch.where(mismatch)[0]:
#                 errors.append({
#                     "index": indices[i].item(),
#                     "true_label": targets[i].item(),
#                     "pred_label": preds[i].item()
#                 })

#     avg_loss = total_loss / total_samples
#     accuracy = total_correct / total_samples
#     return avg_loss, accuracy, errors

# class_names = [
#     "airplane",
#     "bird",
#     "car",
#     "cat",
#     "deer",
#     "dog",
#     "horse",
#     "monkey",
#     "ship",
#     "truck",
# ]

# # --- 準備 ---
# transform = transforms.Compose([
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# # test_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)
# test_dataset = STL10(root="./data", split="train", download=True, transform=transform)
# indexed_test = IndexedDataset(test_dataset)
# test_loader = DataLoader(indexed_test, batch_size=128, shuffle=False)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model = Wide_ResNet(28, 10, 0.3, num_classes=10).to(device)
# model = ResNet18().to(device)
# # model_save_path = "./logs/wide_resnet_28_10/Mixup/cifar10_250_0.pth"
# model_save_path = "./logs/resnet18/Mixup/stl10_200.pth"
# criterion = nn.CrossEntropyLoss()

# model.load_state_dict(torch.load(model_save_path, weights_only=True))
# test_loss, test_acc, misclassified = test_and_collect_errors(model, test_loader, criterion, device)

# print(f"Loss={test_loss:.4f}  Acc={test_acc:.4f}  Errors={len(misclassified)} samples")
# # 最初の数件を表示
# for e in misclassified[:10]:
#     # print(f"  idx={e['index']}  true={e['true_label']}  pred={e['pred_label']}")
#     true_name = class_names[e['true_label']]
#     pred_name = class_names[e['pred_label']]
#     print(f"idx={e['index']}  true={true_name}  pred={pred_name}")


# # 誤分類サンプルの先頭16件
# samples = misclassified[:16]

# fig, axes = plt.subplots(4, 4, figsize=(8,8))
# for ax, s in zip(axes.flatten(), samples):
#     img, _ = test_dataset[s["index"]]     # transform 後の Tensor
#     img = img.numpy().transpose(1,2,0)
#     img = np.clip(img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]), 0, 1)
#     ax.imshow(img)

#     # ← ここでその都度ラベル名を取り出す
#     true_name = class_names[s["true_label"]]
#     pred_name = class_names[s["pred_label"]]
#     ax.set_title(f"T: {true_name} / P: {pred_name}")

#     ax.axis("off")
# plt.tight_layout()
# plt.savefig("error_samples_STL10.png")
# plt.show()


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
from batch_sampler import extract_wrn_features, FeatureKNNBatchSampler, HybridFOMABatchSampler

augmentations = [
    # "Default",
    "Mixup",
    # "Manifold-Mixup",
    # "Local-FOMA",
]

def main():
    for i in range(1):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs",     type=int, default=400)
        parser.add_argument("--data_type",  type=str, default="cifar100",  choices=["stl10", "cifar100", "cifar10"])
        parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "resnet101", "wide_resnet_28_10"])
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
            test_dataset = STL10(root="./data", split="train", download=True, transform=transform)
        elif data_type == "cifar100":
            test_dataset = CIFAR100(root="./data", train=False, transform=transform, download=True)
        elif data_type == "cifar10":
            test_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)

        test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
        
        for augment in augmentations:
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

if __name__ == "__main__":
    main()