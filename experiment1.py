import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.transforms as transforms
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
def compute_jacobian_norm(model, x):
    x = x.requires_grad_(True)
    y = model(x)
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
            out = F.softmax(model(x_mix), dim=1)
            probs.append(out.cpu().numpy())
    return np.stack(probs)  # [len(lambdas), batch, num_classes]

# --- 7. 実験実行サンプル ---
if __name__ == '__main__':

    # 最適化器・損失定義
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Baseline と Mixup で訓練
    for augment in ["Default", "Mixup"]:
        print(f"== {augment} ==")
        model_save_path = f"./logs/wide_resnet_28_10/{augment}/cifar100_400_0.pth"
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.eval()

        # Jacobian ノルム測定
        x_batch, _ = next(iter(test_loader))
        jn = compute_jacobian_norm(model, x_batch.to(device))
        print(f"Avg Jacobian norm ({augment}):", jn.mean().item())

    # 線形補間テスト例
    x_batch, y_batch = next(iter(test_loader))
    # クラス0 とクラス1 の最初のサンプルを選択
    idx0 = (y_batch == 0).nonzero()[0]
    idx1 = (y_batch == 1).nonzero()[0]
    x0 = x_batch[idx0:idx0+1].to(device)
    x1 = x_batch[idx1:idx1+1].to(device)
    lambdas = np.linspace(0, 1, 11)
    probs_baseline = interpolation_test(model, x0, x1, lambdas)
    print("Interpolation probs (Default):", probs_baseline[:,0,:2])
    probs_mixup = interpolation_test(m, x0, x1, lambdas)
    print("Interpolation probs (Mixup):", probs_mixup[:,0,:2])
