import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from src.models.wide_resnet import Wide_ResNet

# --- 1. データセット準備 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = CIFAR100(root="./data", train=False, transform=transform, download=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

def power_iteration(W, num_iters=20):
    # W: 2D tensor
    device = W.device
    u = torch.randn(W.size(0), 1, device=device)
    for _ in range(num_iters):
        v = torch.matmul(W.t(), u)
        v = v / (v.norm() + 1e-12)
        u = torch.matmul(W, v)
        u = u / (u.norm() + 1e-12)
    sigma = torch.matmul(u.t(), torch.matmul(W, v))
    return sigma.item()

# --- 2. スペクトルノルム＆Lipschitz定数（近似）の計算 ---
def compute_layer_spectral_norms(model):
    norms = []
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            W = param.view(param.size(0), -1)
            sigma = power_iteration(W)
            norms.append(sigma)
    return norms

# --- 3. 摂動耐性テスト ---
def perturbation_test(model, x, device, noise_std=0.01):
    x = x.to(model.device)
    delta = torch.randn_like(x) * noise_std
    x_noisy = x + delta
    with torch.no_grad():
        out_clean = F.softmax(model(x, labels=None, device=device, augment=None), dim=1)
        out_noisy = F.softmax(model(x_noisy, labels=None, device=device, augment=None), dim=1)
    diff = (out_noisy - out_clean).view(x.size(0), -1).norm(dim=1)
    delta_norm = delta.view(x.size(0), -1).norm(dim=1)
    ratio = (diff / delta_norm).cpu().numpy()
    return ratio

# --- 4. 実験実行 ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    augmentations = ['Default', 'Mixup', 'Manifold-Mixup']
    for aug in augmentations:
        # モデル読み込み
        model = Wide_ResNet(28, 10, 0.3, num_classes=100).to(device)
        ckpt = f'./logs/wide_resnet_28_10/{aug}/cifar100_400_0.pth'
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        model.eval()
        model.device = device

        # 2a. スペクトルノルム計算
        norms = compute_layer_spectral_norms(model)
        lipschitz_est = np.prod(norms)
        print(f"[{aug}] Spectral norms per layer: {np.round(norms, 4)}")
        print(f"[{aug}] Approx. Lipschitz constant (product): {lipschitz_est:.4e}")

        # 2b. 摂動耐性テスト
        x_batch, _ = next(iter(test_loader))
        ratios = perturbation_test(model, x_batch, device, noise_std=0.01)
        print(f"[{aug}] Perturbation ratio mean: {ratios.mean():.4f}, std: {ratios.std():.4f}\n")
