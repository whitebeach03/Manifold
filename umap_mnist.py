import umap
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.datasets import CIFAR10

# MNIST を NumPy 配列で取得
mnist = fetch_openml(
    'mnist_784',
    version=1,
    as_frame=False,
    parser='liac-arff'
)
X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)

# サンプル絞り込み
rng = np.random.RandomState(42)
idx = rng.choice(len(X), 6000, replace=False)
X_sub, y_sub = X[idx], y[idx]

# trainset = CIFAR10(root='./data', train=True, download=True)
# # NumPy 配列に変換して (N, 32*32*3) のベクトルにリシェイプ＋0–1 正規化
# X = trainset.data.astype(np.float32).reshape(len(trainset), -1) / 255.0
# y = np.array(trainset.targets)   # (N,) のラベル配列

# サンプル数を絞る（例：2,000件）
rng = np.random.RandomState(42)
idx = rng.choice(len(X), 7000, replace=False)
X_sub, y_sub = X[idx], y[idx]

# UMAP 次元削減（警告は無視してOKです）
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_sub)

# プロット
plt.figure(figsize=(8, 8))
cmap = cm.get_cmap('tab10', 10)
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=y_sub,
    cmap=cmap,
    s=5,
    alpha=0.8
)

# 凡例：handles だけ取得して、ラベルは自分で用意
handles, _ = scatter.legend_elements(prop="colors", alpha=0.8, num=10)
labels = [str(i) for i in range(10)]
plt.legend(handles, labels, title="Digit", loc="best", fontsize="small")

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of MNIST')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig("UMAP_MNIST.png")
plt.show()