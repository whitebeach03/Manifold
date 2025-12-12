# import matplotlib.pyplot as plt
# from sklearn import datasets
# import numpy as np

# # スイスロールデータの生成
# n_samples = 2000
# noise = 0.05
# X, t = datasets.make_swiss_roll(n_samples=n_samples, noise=noise)

# # プロットの準備
# fig = plt.figure(figsize=(12, 5))

# # 1. 高次元空間（3D）のプロット：入力空間のメタファー
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.Spectral, s=10)
# ax1.set_title("Input Space (High-dimensional)", fontsize=14)
# ax1.set_xlabel("Pixel x")
# ax1.set_ylabel("Pixel y")
# ax1.set_zlabel("Pixel z")
# ax1.view_init(10, -70)

# # 2. 低次元多様体（2D）のプロット：特徴空間のメタファー
# # スイスロールを開いた状態（tとy座標を使用）
# ax2 = fig.add_subplot(122)
# ax2.scatter(t, X[:, 1], c=t, cmap=plt.cm.Spectral, s=10)
# ax2.set_title("Intrinsic Manifold (Low-dimensional)", fontsize=14)
# ax2.set_xlabel("Manifold Coordinate 1")
# ax2.set_ylabel("Manifold Coordinate 2")
# ax2.grid(True, linestyle='--', alpha=0.6)

# plt.tight_layout()
# plt.savefig("manifold_hypothesis.png", dpi=300)
# plt.show()



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 再現性のためのシード固定
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. データセットの生成 (Two Moons)
# ==========================================
# ノイズを少し多めにして、過学習しやすい状況を作る
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# PyTorchのTensorに変換
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# ==========================================
# 2. ニューラルネットワークモデルの定義
# ==========================================
# 比較的高次元の中間層を持たせ、過学習しやすくする
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

# ==========================================
# 3. 学習関数の定義
# ==========================================
def train_model(model, optimizer, criterion, X, y, use_mixup=False, alpha=1.0, epochs=10000):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if use_mixup:
            # Mixupの実装
            # ランダムなインデックスを生成してシャッフル
            indices = torch.randperm(X.size(0))
            x1 = X
            y1 = y
            x2 = X[indices]
            y2 = y[indices]
            
            # Beta分布からlambdaをサンプリング
            lam = np.random.beta(alpha, alpha)
            
            # 入力の混合
            x_mixed = lam * x1 + (1 - lam) * x2
            
            # 出力の計算
            outputs = model(x_mixed)
            
            # 損失の計算（ラベルも混合比率で重み付け）
            loss = lam * criterion(outputs, y1) + (1 - lam) * criterion(outputs, y2)
            
        else:
            # 通常の学習 (ERM)
            outputs = model(X)
            loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()

# ==========================================
# 4. モデルの学習と描画
# ==========================================

# グリッドの作成（描画用）
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 設定：左(Standard), 右(Mixup)
settings = [
    {"title": "Standard Training", "mixup": False},
    {"title": "Mixup Training (alpha=1.0)", "mixup": True}
]

for i, setting in enumerate(settings):
    # モデルの初期化
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 学習実行
    print(f"Training {setting['title']}...")
    train_model(model, optimizer, criterion, X_tensor, y_tensor, 
                use_mixup=setting['mixup'], epochs=2000)
    
    # 予測（等高線描画用）
    model.eval()
    with torch.no_grad():
        Z = model(grid_tensor)
        Z = torch.softmax(Z, dim=1)[:, 1] # クラス1の確率を取得
        Z = Z.reshape(xx.shape).numpy()
    
    # プロット
    ax = axes[i]
    contour = ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 21))
    
    # データ点のプロット
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', edgecolors='white', s=60, label='Class 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', edgecolors='white', s=60, label='Class 1')
    
    ax.set_title(setting['title'], fontsize=16)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.savefig("real_decision_boundary.png", dpi=300)
print("Saved real_decision_boundary.png")
plt.show()


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_circles

# # シード固定
# torch.manual_seed(42)
# np.random.seed(42)

# # ==========================================
# # 1. データセットの生成 (同心円: Make Circles)
# # ==========================================
# # noise=0.1 で少し重なりを持たせ、難易度を上げる
# X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

# X_tensor = torch.FloatTensor(X)
# y_tensor = torch.LongTensor(y)

# # ==========================================
# # 2. モデル定義 (少し深めのMLP)
# # ==========================================
# class DeepMLP(nn.Module):
#     def __init__(self):
#         super(DeepMLP, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(2, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 2)
#         )
    
#     def forward(self, x):
#         return self.layers(x)

# # ==========================================
# # 3. 学習ループ (Standard vs Mixup)
# # ==========================================
# def train_experiment(use_mixup, epochs=20000):
#     model = DeepMLP()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#     criterion = nn.CrossEntropyLoss()
    
#     model.train()
#     for _ in range(epochs):
#         optimizer.zero_grad()
        
#         if use_mixup:
#             # Mixup
#             indices = torch.randperm(X_tensor.size(0))
#             x1, y1 = X_tensor, y_tensor
#             x2, y2 = X_tensor[indices], y_tensor[indices]
            
#             lam = np.random.beta(1.0, 1.0)
#             x_mixed = lam * x1 + (1 - lam) * x2
#             outputs = model(x_mixed)
#             loss = lam * criterion(outputs, y1) + (1 - lam) * criterion(outputs, y2)
#         else:
#             # Standard
#             outputs = model(X_tensor)
#             loss = criterion(outputs, y_tensor)
            
#         loss.backward()
#         optimizer.step()
#     return model

# # ==========================================
# # 4. 描画
# # ==========================================
# # グリッド作成
# x_range = np.linspace(-1.5, 1.5, 200)
# y_range = np.linspace(-1.5, 1.5, 200)
# xx, yy = np.meshgrid(x_range, y_range)
# grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

# fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
# titles = ["Standard Training (ERM)", "Mixup Training"]
# mixup_flags = [False, True]

# for i, (ax, use_mixup) in enumerate(zip(axes, mixup_flags)):
#     print(f"Training {titles[i]}...")
#     model = train_experiment(use_mixup)
    
#     # 予測
#     model.eval()
#     with torch.no_grad():
#         preds = torch.softmax(model(grid_tensor), dim=1)[:, 1]
#         Z = preds.reshape(xx.shape).numpy()
    
#     # 等高線 (確率のグラデーション)
#     contour = ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.8)
    
#     # データ点
#     ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', edgecolors='white', s=40, label='Class 0 (Outer)')
#     ax.scatter(X[y==1, 0], X[y==1, 1], c='red', edgecolors='white', s=40, label='Class 1 (Inner)')
    
#     ax.set_title(titles[i], fontsize=14)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     if i == 0: ax.legend(loc='upper right')

# plt.tight_layout()
# plt.savefig("circles_decision_boundary.png", dpi=300)
# print("Saved circles_decision_boundary.png")
# plt.show()