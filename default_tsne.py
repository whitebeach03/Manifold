# import random
# import numpy as np
# import torch
# from torchvision import datasets, transforms
# import umap  # pip install umap-learn
# import matplotlib.pyplot as plt

# # ———— Cutout の実装 ————
# class Cutout(object):
#     def __init__(self, n_holes, length):
#         self.n_holes = n_holes
#         self.length = length

#     def __call__(self, img):
#         h = img.size(1)
#         w = img.size(2)

#         mask = np.ones((h, w), np.float32)

#         for _ in range(self.n_holes):
#             y = random.randint(0, h - 1)
#             x = random.randint(0, w - 1)

#             y1 = np.clip(y - self.length // 2, 0, h)
#             y2 = np.clip(y + self.length // 2, 0, h)
#             x1 = np.clip(x - self.length // 2, 0, w)
#             x2 = np.clip(x + self.length // 2, 0, w)

#             mask[y1: y2, x1: x2] = 0.

#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img = img * mask

#         return img

# # ———— 変換定義 ————
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406],
#                          std=[0.229,0.224,0.225])
# ])

# default_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Pad(4),
#     transforms.RandomCrop(32),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406],
#                          std=[0.229,0.224,0.225]),
#     # Cutout(n_holes=1, length=16),
# ])

# # ———— データ読み込み ————
# train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)

# # 比較するクラス番号（例：0）
# for i in range(10):
#     class_idx = i
#     all_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_idx]

#     # サンプル数
#     sample_size = 500
#     random.seed(0)
#     sample_indices = random.sample(all_indices, sample_size)

#     # ———— 特徴ベクトル作成 ————
#     orig_vecs = []
#     aug_vecs  = []
#     for idx in sample_indices:
#         img, _ = train_dataset[idx]
#         orig_vecs.append(transform(img).view(-1).numpy())
#         aug_vecs.append(default_transform(img).view(-1).numpy())

#     X = np.vstack(orig_vecs + aug_vecs)

#     # ———— UMAP 実行 ————
#     reducer = umap.UMAP(
#         n_components=2,
#         n_neighbors=15,     # 近傍点数。データの局所構造に応じて調整してください
#         min_dist=0.1,       # 埋め込み後の点同士の最小距離
#         metric='euclidean',
#         random_state=0
#     )
#     X2 = reducer.fit_transform(X)

#     # 分割
#     X_orig = X2[:sample_size]
#     X_aug  = X2[sample_size:]

#     # ———— プロット ————
#     plt.figure(figsize=(8,8))
#     plt.scatter(X_orig[:,0], X_orig[:,1], label='Original',          alpha=0.6)
#     plt.scatter(X_aug[:,0],  X_aug[:,1],  label='Default Transform', alpha=0.6)
#     plt.legend()
#     plt.title('UMAP: CIFAR-100 Class {} (Original vs Default)'.format(class_idx))
#     plt.xlabel('UMAP 1')
#     plt.ylabel('UMAP 2')
#     plt.savefig(f"original-default-{i}.png")

from src.models.wide_resnet import Wide_ResNet
import random
import numpy as np
import torch
from torchvision import datasets, transforms
import umap           # pip install umap-learn
import matplotlib.pyplot as plt


# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのロード
num_classes = 100
model = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=num_classes).to(device)
model_save_path = "./logs/wide_resnet_28_10/Default/cifar100_400_0.pth"
state = torch.load(model_save_path, map_location=device)
model.load_state_dict(state)
model.eval()

# 変換定義
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

default_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# データ読み込み
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)

# 各クラス0-9でUMAPを実行、500サンプルずつ抽出して保存
sample_size = 500
random.seed(0)

for class_idx in range(10):
    # クラスごとのインデックス取得
    all_indices = [i for i, (_, lbl) in enumerate(train_dataset) if lbl == class_idx]
    sample_indices = random.sample(all_indices, sample_size)

    # 特徴抽出
    orig_feats = []
    aug_feats  = []
    with torch.no_grad():
        for idx in sample_indices:
            img, _ = train_dataset[idx]
            # 元画像特徴
            x_orig = transform(img).unsqueeze(0).to(device)
            f_orig = model.extract_features(x_orig)
            orig_feats.append(f_orig.cpu().numpy().squeeze())
            # 拡張画像特徴
            x_aug = default_transform(img).unsqueeze(0).to(device)
            f_aug = model.extract_features(x_aug)
            aug_feats.append(f_aug.cpu().numpy().squeeze())

    # UMAP圧縮
    X = np.vstack((orig_feats, aug_feats))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=0
    )
    X2 = reducer.fit_transform(X)
    X_orig, X_aug = X2[:sample_size], X2[sample_size:]

    # プロットを保存
    plt.figure(figsize=(8,8))
    plt.scatter(X_orig[:,0], X_orig[:,1], label='Original Features', alpha=0.6)
    plt.scatter(X_aug[:,0],  X_aug[:,1],  label='Augmented Features', alpha=0.6)
    plt.legend()
    plt.title(f'UMAP on Wide-ResNet Features (Class {class_idx})')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(f"umap_features_class_{class_idx}.png")
    plt.close()
