import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= 簡易CNNモデル（特徴抽出部分を定義） =========

class SimpleCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # CIFAR10前提
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, feature_dim)
        )

    def forward(self, x):
        return self.features(x)



def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes).float()


def step1(model, x_batch, y_batch, num_classes):
    """
    ステップ1: 特徴とone-hotラベルの結合
    model: 特徴抽出モデル
    x_batch: 入力画像 (B, C, H, W)
    y_batch: 整数ラベル (B,)
    num_classes: クラス数
    """
    model.eval()
    with torch.no_grad():
        z_l = model(x_batch)  # 中間特徴 (B, n_l)
        y_one_hot = one_hot(y_batch, num_classes)  # ラベルone-hot (B, C)
        A = torch.cat([z_l, y_one_hot], dim=1)  # 結合 (B, n_l + C)
    return A

def step2(A, k, lam):
    """
    ステップ2: SVDを行い、下位特異値をλでスケールして新しいAを再構成する
    A: ステップ1で得た行列 (B, n_l + C)
    lam: λ ∈ [0, 1] (float) 特異値スケーリング係数
    k: スケーリングせずに残す上位特異値の個数 (int)
    """
    # フルSVD: A = U S V^T
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)  
    lam_repeat = lam.repeat(S.shape[-1] - k)
    lam_ones = torch.ones(k)
    S = S * torch.cat((lam_ones, lam_repeat))
    A = U @ torch.diag(S) @ Vt
    return A

def step3(A, feature_dim, num_classes):
    """
    ステップ3: 再構成されたAを特徴部分とラベル部分に分離する
    A: SVDスケーリング後のA (B, feature_dim + num_classes)
    feature_dim: 特徴次元 n_l
    num_classes: クラス数 C
    """
    # 特徴部分（左側の n_l 列）
    Z_l_aug = A[:, :feature_dim]

    # ラベル部分（右側の C 列）
    Y_soft = A[:, feature_dim:]

    return Z_l_aug, Y_soft


# ========= テストコード =========

if __name__ == "__main__":
    # 仮データの準備
    batch_size = 128
    num_classes = 10
    feature_dim = 512

    # ランダム画像とラベル（CIFAR10風：3x32x32）
    x_dummy = torch.randn(batch_size, 3, 32, 32)
    y_dummy = torch.randint(0, num_classes, (batch_size,))

    # モデルのインスタンス化
    model = SimpleCNN(feature_dim=feature_dim)

    # ステップ1の実行
    A = step1(model, x_dummy, y_dummy, num_classes)

    # ステップ2の実行
    lam = torch.Tensor([0.5]) 
    k = 20
    A = step2(A, k, lam)

    # ステップ3の実行
    Z_l_aug, Y_soft = step3(A, feature_dim, num_classes)
    print(Z_l_aug.shape)
    print(Y_soft)