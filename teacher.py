import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from src.models.wide_resnet import Wide_ResNet
from src.methods.mixup import *
import matplotlib.pyplot as plt

# --- 1. 準備 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 の前処理
default_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# データセット（ダウンロード済みが前提）
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=default_transform, download=False)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# --- 3. モデル読み込み ---
model = Wide_ResNet(28, 10, 0.3, num_classes=10).to(device)

# 必要に応じて訓練済み重みをロード
model.load_state_dict(torch.load("./logs/wide_resnet_28_10/Original/cifar10_250_0.pth", weights_only=True))
model.eval()

# --- 4. １バッチ取得して mixup し、予測確率を表示 ---
images, labels = next(iter(loader))  # images.shape == [2,3,32,32]
images, labels = images.to(device), labels.to(device)

mixup_fn   = Mixup(alpha=1.0, mode="batch", num_classes=10)
lam, index = mixup_fn._get_params(images.size(0), device)
mixed_images = mixup_fn._linear_mixing(lam, images, index)

with torch.no_grad():
    probs = model(mixed_images, labels=labels, device=device, augment="Mixup")             # [2,10]
    probs = F.softmax(probs, dim=1)         # [2,10]

classes = dataset.classes  # CIFAR-10 のラベル名リスト

# 7. 逆正規化用関数（表示用）
inv_norm = transforms.Normalize(
    mean=[-m/s for m, s in zip((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))],
    std=[1/s for s in (0.2023,0.1994,0.2010)]
)

# 結果表示
for i in range(mixed_images.size(0)):
    print(f"-- Sample {i} (λ = {lam:.3f}) --")
    # 混合元ラベル
    print(" Original labels:", labels[i].item(), " & ",
          labels[torch.randperm(2)[i]].item())
    # ネットワークの予測確率
    for cls in range(10):
        print(f"  Class {cls:2d}: {probs[i, cls].item():.4f}")
    print()


# 8. 表示
# 今回はバッチサイズ=2なので、i=0 の例を一つだけ描画
i = 0
j = index[i].item()  # mixup の相手インデックス

fig, axes = plt.subplots(1, 3, figsize=(12,4))

# 左: 元画像 A
imgA = inv_norm(images[i].cpu()).permute(1,2,0).numpy().clip(0,1)
axes[0].imshow(imgA)
axes[0].axis('off')
axes[0].set_title(f"A: {classes[labels[i].item()]}")

# 中央: 元画像 B
imgB = inv_norm(images[j].cpu()).permute(1,2,0).numpy().clip(0,1)
axes[1].imshow(imgB)
axes[1].axis('off')
axes[1].set_title(f"B: {classes[labels[j].item()]}")

# 右: Mixup 画像
imgM = inv_norm(mixed_images[i].cpu()).permute(1,2,0).numpy().clip(0,1)
axes[2].imshow(imgM)
axes[2].axis('off')
axes[2].set_title(f"Mixup (λ={lam:.2f}, 1−λ={1-lam:.2f})")

plt.tight_layout()
plt.savefig("mixup.png")

# 9. 棒グラフで予測確率
plt.figure(figsize=(8,4))
plt.bar(range(10), probs[i].cpu().numpy())
plt.xticks(range(10), classes, rotation=45)
plt.xlabel("Class")
plt.ylabel("Probability")
plt.title("Predicted Probabilities for the Mixup Image")
plt.tight_layout()
plt.savefig("mixup_graph.png")