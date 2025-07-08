import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, STL10
import torchvision.transforms as transforms
from src.models.wide_resnet import Wide_ResNet
from src.models.resnet import ResNet18
from tqdm import tqdm

#── 1) IndexedDataset ────────────────────────────────────────────
class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx

#── 2) Mixup (per-sample λ) ────────────────────────────────────────
def mixup_data(x, y, idx, alpha=1.0, device="cuda"):
    """
    x: [B,C,H,W] tensor
    y, idx: [B]
    Returns:
      mixed_x: [B,C,H,W]
      y_a, y_b, idx_a, idx_b: [B]
      lam: [B]  ← per-sample mixing coefficient
    """
    B = x.size(0)
    if alpha <= 0:
        lam = torch.ones(B, device=device)
        return x, y, y, idx, idx, lam

    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)
    perm = torch.randperm(B, device=device)
    lam_x = lam.view(B,1,1,1)  # broadcast

    mixed_x = lam_x * x + (1 - lam_x) * x[perm]
    y_a, y_b   = y,    y[perm]
    idx_a, idx_b = idx,  idx[perm]

    return mixed_x, y_a, y_b, idx_a, idx_b, lam

#── 3) テスト＋誤分類サンプル収集 ───────────────────────────────────
def test_mixup_and_collect_errors(model, loader, criterion, device, alpha=1.0):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    errors = []

    with torch.no_grad():
        for inputs, targets, indices in tqdm(loader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            indices = indices.to(device)

            mixed_x, y_a, y_b, ia, ib, lam = mixup_data(inputs, targets, indices, alpha, device)
            outputs = model(mixed_x, labels=targets, device=device, augment="Mixup")

            # per-sample loss
            loss_a = F.cross_entropy(outputs, y_a, reduction='none')
            loss_b = F.cross_entropy(outputs, y_b, reduction='none')
            loss = (lam * loss_a + (1-lam) * loss_b).mean()
            total_loss += (lam * loss_a + (1-lam) * loss_b).sum().item()

            # Mixup 精度（擬似）
            preds = outputs.argmax(dim=1)
            correct = ((preds == y_a).float() * lam + (preds == y_b).float() * (1-lam)).sum().item()
            total_correct += correct
            total_samples += lam.size(0)

            # 誤分類サンプルだけ集める
            mismatch = ~((preds == y_a) | (preds == y_b))
            for i in torch.where(mismatch)[0]:
                errors.append({
                    "idx_a": ia[i].item(),
                    "idx_b": ib[i].item(),
                    "true_a": y_a[i].item(),
                    "true_b": y_b[i].item(),
                    "pred": preds[i].item(),
                    "lam": lam[i].item()
                })

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, errors

#── 4) 可視化 ────────────────────────────────────────────────────
def denormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1,2,0)
    return np.clip(
        img * np.array([0.229,0.224,0.225]) +
        np.array([0.485,0.456,0.406]),
        0,1
    )

# def visualize_errors(i, test_base, errors, class_names, num_images=16):
#     # 実際に表示する数
#     num_disp = min(num_images, len(errors))
#     samples = errors[:num_disp]

#     # 列数は最大4列、それ以上なら4列
#     cols = min(4, num_disp)
#     # 行数は切り上げ
#     rows = math.ceil(num_disp / cols)

#     fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
#     # axes が 1D or 2D どちらでも flatten して使えるように
#     axes = np.array(axes).reshape(-1)

#     for ax, e in zip(axes, samples):
#         img_a, _ = test_base[e["idx_a"]]
#         img_b, _ = test_base[e["idx_b"]]
#         lam = e["lam"]
#         mixed = lam * img_a + (1-lam) * img_b

#         ax.imshow(denormalize(mixed))
#         ax.set_title(
#             f"{class_names[e['true_a']]}:{lam:.2f}\n"
#             f"{class_names[e['true_b']]}:{1-lam:.2f}\n"
#             f"→ {class_names[e['pred']]}",
#             fontsize=10
#         )
#         ax.axis("off")

#     # 余ったサブプロットは消す
#     for ax in axes[num_disp:]:
#         ax.axis("off")

#     plt.tight_layout()
#     plt.savefig(f"error_mix_stl10_{i}.png")
#     plt.show()

def visualize_errors(i, test_base, errors, class_names, num_images=16):
    """
    i:       ページ番号 (0 オリジン)
    test_base: IndexedDataset ではなく元の dataset (正規化済みテンソルを返すもの)
    errors:  誤分類リスト (len(errors) >= 1)
    class_names: ラベル名リスト
    num_images: 1 回に表示する枚数
    """
    # 1) チャンクを切り出し
    start = i * num_images
    end   = start + num_images
    samples = errors[start:end]
    num_disp = len(samples)
    if num_disp == 0:
        print(f"[visualize_errors] ページ {i} には表示すべきサンプルがありません。")
        return

    # 2) レイアウト計算
    cols = min(4, num_disp)
    rows = math.ceil(num_disp / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).reshape(-1)

    # 3) 各サンプルを描画
    for ax, e in zip(axes, samples):
        img_a, _ = test_base[e["idx_a"]]
        img_b, _ = test_base[e["idx_b"]]
        lam = e["lam"]
        mixed = lam * img_a + (1 - lam) * img_b

        ax.imshow(denormalize(mixed))
        ax.set_title(
            f"{class_names[e['true_a']]}:{lam:.2f}\n"
            f"{class_names[e['true_b']]}:{(1-lam):.2f}\n"
            f"→ {class_names[e['pred']]}",
            fontsize=10
        )
        ax.axis("off")

    # 4) 余剰サブプロットをオフに
    for ax in axes[num_disp:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"error_mix_stl10_{i}.png")
    plt.show()

#── 5) 実行例 ────────────────────────────────────────────────────
data_type = "stl10"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485,0.456,0.406],
      std =[0.229,0.224,0.225]
    ),
])

if data_type == "stl10":
    model_save_path = "./logs/resnet18/Mixup/stl10_200.pth"
    test_base = STL10(root="./data", split="train", download=True, transform=transform)
    model = ResNet18().to(device)

elif data_type == "cifar10":
    model_save_path = "./logs/wide_resnet_28_10/Mixup/cifar10_250_0.pth"
    test_base = CIFAR10(root="./data", train=False, transform=transform, download=True)
    model = Wide_ResNet(28, 10, 0.3, num_classes=10).to(device)

test_dataset = IndexedDataset(test_base)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

model.load_state_dict(torch.load(model_save_path, weights_only=True))

criterion = nn.CrossEntropyLoss()

loss, acc, errors = test_mixup_and_collect_errors(
    model, test_loader, criterion, device, alpha=0.3
)
print(f"Mixup-Test Loss: {loss:.4f}, Acc: {acc:.4f}, Errors: {len(errors)}件")

for i in range(5):
    visualize_errors(i, test_base, errors, test_base.classes, num_images=16)
