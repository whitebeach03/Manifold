import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from src.models.wide_resnet import Wide_ResNet
from src.utils import mixup_data

# --- 1. データセット準備 ---
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = CIFAR100(root="./data", train=False, transform=transform, download=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# --- 2. モデル定義 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Wide_ResNet(28, 10, 0.3, num_classes=100).to(device)

# --- 5. Jacobian ノルム測定関数 ---
def compute_jacobian_norm(model, x, labels, augment):
    x = x.requires_grad_(True)
    y = model(x, labels, device, augment)
    # クラスごとに sum outputs, compute grad w.r.t. input
    grads = []
    for i in range(y.size(1)):
        model.zero_grad()
        grad_i = torch.autograd.grad(outputs=y[:, i].sum(), inputs=x, retain_graph=True)[0]
        grads.append(grad_i.view(grad_i.size(0), -1))
    J = torch.stack(grads, dim=2)  # [batch, input_dim, num_classes]
    frob_norms = torch.norm(J, dim=(1,2))  # 各サンプルの Frobenius ノルム
    return frob_norms

# --- 6. 線形補間テスト関数 ---
def interpolation_test(model, x1, x2, lambdas):
    model.eval()
    probs = []
    with torch.no_grad():
        for lam in lambdas:
            x_mix = lam * x1 + (1 - lam) * x2
            out = F.softmax(model(x_mix, labels=None, device=None, augment=None), dim=1)
            probs.append(out.cpu().numpy())
    return np.stack(probs)  # [len(lambdas), batch, num_classes]

def plot(augmentations, lambdas, apple1, apple2, apple3, wolf1, wolf2, wolf3):
    os.makedirs(f"./experiments/", exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    ### sample 1 ###
    plt.subplot(1, 2, 1)
    plt.plot(lambdas, apple1, linestyle='solid', linewidth=0.8, label="Default")
    plt.plot(lambdas, apple2, linestyle='solid', linewidth=0.8, label="Mixup")
    plt.plot(lambdas, apple3, linestyle='solid', linewidth=0.8, label="Manifold-Mixup")
        
    plt.title('Apple Probs.')
    plt.xlabel('λ')
    plt.ylabel('Probs.')
    plt.legend()
    plt.grid(True)
    # plt.ylim(bottom=0.2)
    
    ### sample 2 ###
    plt.subplot(1, 2, 2)
    plt.plot(lambdas, wolf1, linestyle='solid', linewidth=0.8, label="Default")
    plt.plot(lambdas, wolf2, linestyle='solid', linewidth=0.8, label="Mixup")
    plt.plot(lambdas, wolf3, linestyle='solid', linewidth=0.8, label="Manifold-Mixup")
        
    plt.title('Wolf Probs.')
    plt.xlabel('λ')
    plt.ylabel('Probs.')
    plt.legend()
    plt.grid(True)
    # plt.ylim(bottom=0.2)
    
    plt.tight_layout()
    plt.savefig(f'./experiments/Default_Mixup.png')
    print("Save Result!")

# --- 7. 実験実行サンプル ---
if __name__ == '__main__':
    augmentations = ["Default", "Mixup", "Manifold-Mixup"]
    default_apples = []
    default_wolfs  = []
    mixup_apples = []
    mixup_wolfs  = []
    manifold_mixup_apples = []
    manifold_mixup_wolfs  = []

    # Baseline と Mixup で訓練
    for augment in augmentations:
        print(f"== {augment} ==")
        model_save_path = f"./logs/wide_resnet_28_10/{augment}/cifar100_400_0.pth"
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.eval()

        # Jacobian ノルム測定
        x_batch, labels = next(iter(test_loader))
        jn = compute_jacobian_norm(model, x_batch.to(device), labels.to(device), augment)
        print(f"Avg Jacobian norm ({augment}):", jn.mean().item())

        # 線形補間テスト例
        x_batch, y_batch = next(iter(test_loader))
        # クラス0 とクラス1 の最初のサンプルを選択
        idx0 = (y_batch == 21).nonzero()[0] # チンパンジー
        idx1 = (y_batch == 97).nonzero()[0] # オオカミ
        x0 = x_batch[idx0:idx0+1].to(device)
        x1 = x_batch[idx1:idx1+1].to(device)
        lambdas = np.linspace(0, 1, 101)
        probs_baseline = interpolation_test(model, x0, x1, lambdas)
        print(f"Interpolation probs ({augment}):")
        
        for i in range(101):
            apple_prob = probs_baseline[i][0][21] 
            wolf_prob  = probs_baseline[i][0][97]
            if augment == "Default":
                default_apples.append(apple_prob)
                default_wolfs.append(wolf_prob)
            elif augment == "Mixup":
                mixup_apples.append(apple_prob)
                mixup_wolfs.append(wolf_prob)
            elif augment == "Manifold-Mixup":
                manifold_mixup_apples.append(apple_prob)
                manifold_mixup_wolfs.append(wolf_prob)
            print(round(apple_prob, 3), round(wolf_prob, 3))

    plot(augmentations, lambdas, default_apples, mixup_apples, manifold_mixup_apples, default_wolfs, mixup_wolfs, manifold_mixup_wolfs)