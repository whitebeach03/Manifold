# import umap
# import torchvision.transforms as transforms
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from torchvision.datasets import CIFAR10, STL10
# from sklearn.datasets import fetch_openml
# from src.models.resnet import ResNet18

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#       mean=[0.485,0.456,0.406],
#       std =[0.229,0.224,0.225]
#     ),
# ])

# model = ResNet18().to(device)
# model.eval()

# # trainset = CIFAR10(root='./data', train=True, download=True)
# trainset = STL10(root="./data", split="train", download=True, transform=transform)


# # NumPy 配列に変換して (N, 32*32*3) のベクトルにリシェイプ＋0–1 正規化
# X = trainset.data.astype(np.float32).reshape(len(trainset), -1) / 255.0
# y = np.array(trainset.labels)   # (N,) のラベル配列

# # サンプル数を絞る（例：2,000件）
# rng = np.random.RandomState(42)
# idx = rng.choice(len(X), 3000, replace=False)
# X_sub, y_sub = X[idx], y[idx]

# # UMAP 次元削減（警告は無視してOKです）
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
# embedding = reducer.fit_transform(X_sub)

# # プロット
# plt.figure(figsize=(8, 8))
# cmap = cm.get_cmap('tab10', 10)
# scatter = plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=y_sub,
#     cmap=cmap,
#     s=5,
#     alpha=0.8
# )

# # 凡例：handles だけ取得して、ラベルは自分で用意
# handles, _ = scatter.legend_elements(prop="colors", alpha=0.8, num=10)
# labels = [str(i) for i in range(10)]
# plt.legend(handles, labels, title="Digit", loc="best", fontsize="small")

# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of MNIST (2,000 samples)')
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')
# plt.savefig("UMAP_MNIST.png")
# plt.show()


import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import torchvision.transforms as transforms
from src.models.resnet import ResNet18

# 1) デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) データセット＆DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4467,0.4398,0.4066],
        std =[0.2241,0.2210,0.2239]
    ),
])
ds = STL10(root="./data", split="train", download=True, transform=transform)
loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# 3) ResNet18 の読み込み（学習済み or ファインチューニング済みモデルをロード）
model = ResNet18().to(device)
model_save_path = "./logs/resnet18/Original/stl10_200.pth"
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model = model.to(device)
model.eval()

# 4) 特徴抽出ループ
features_list = []
labels_list   = []
with torch.no_grad():
    for imgs, targets in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        feat = model.extract_features(imgs)
        
        features_list.append(feat.cpu().numpy())
        labels_list.append(targets.numpy())

# 5) NumPy に結合＆サブサンプリング
features = np.concatenate(features_list, axis=0)
labels   = np.concatenate(labels_list,   axis=0)

# ランダムに 3,000 サンプルだけ抽出
rng = np.random.RandomState(42)
idx = rng.choice(len(features), 5000, replace=False)
X_sub, y_sub = features[idx], labels[idx]

# 6) UMAP 次元削減
reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_sub)

# 7) プロット
plt.figure(figsize=(8,8))
cmap = cm.get_cmap('tab10', 10)
scatter = plt.scatter(
    embedding[:,0],
    embedding[:,1],
    c=y_sub,
    cmap=cmap,
    s=5,
    alpha=0.8
)

# 凡例
handles, _ = scatter.legend_elements(prop="colors", alpha=0.8, num=10)
plt.legend(handles, ds.classes, title="Class", loc="best", fontsize="small")

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP of STL10 feature space (Random 3,000 samples)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.tight_layout()
plt.savefig("UMAP_STL10_Original_features.png")
plt.show()
