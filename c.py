import random
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from foma import foma

# ──── Default Transform 定義 (STL-10用) ────
default_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(96, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# ──── データロード ────
train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=None)
device = torch.device('cpu')
num_classes = 10

# ──── 正規化解除ヘルパー ────
def unnormalize(tensor):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return torch.clamp(tensor * std + mean, 0, 1)

# ──── 画像抽出＆FOMA適用 ────
random.seed(0)
orig_imgs, foma_imgs = [], []
for cls in range(num_classes):
    idxs = [i for i,(_,lbl) in enumerate(train_dataset) if lbl==cls]
    idx = random.choice(idxs)
    img, _ = train_dataset[idx]

    x_def = default_transform(img).unsqueeze(0).to(device)
    x_foma, _ = foma(x_def, torch.tensor([cls], device=device),
                     num_classes=num_classes, alpha=1.0, rho=0.9)

    orig_imgs.append(unnormalize(x_def.squeeze(0).cpu()))
    foma_imgs.append(unnormalize(x_foma.squeeze(0).cpu()))

# ──── グリッド描画＆保存 ────
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    axes[0, i].imshow(orig_imgs[i].permute(1,2,0))
    axes[0, i].axis('off')
    axes[0, i].set_title(f'Class {i}\nDefault', fontsize=8)

    axes[1, i].imshow(foma_imgs[i].permute(1,2,0))
    axes[1, i].axis('off')
    axes[1, i].set_title('FOMA', fontsize=8)

plt.tight_layout()
plt.savefig("stl10_default_vs_foma_2x10.png")