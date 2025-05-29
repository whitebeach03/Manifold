import torch
import torch.nn as nn
import torch.nn.functional as F

# 引数設定
class Args:
    num_classes = 10
    alpha = 1.0
    rho = 0.9
    small_singular = True



def foma(X, Y, num_classes, alpha, rho, small_singular=True, lam=None):
    """
    FOMA for image classification tasks.
    X: Input images, shape [B, C, H, W]
    Y: Labels, shape [B] or [B, num_classes]
    """
    B = X.shape[0]
    # Flatten image to [B, C*H*W]
    X_flat = X.view(B, -1)

    # Convert labels to one-hot if needed
    if Y.ndim == 1:  # [B]
        Y_onehot = F.one_hot(Y, num_classes=num_classes).float()
    else:
        Y_onehot = Y.float()

    # Concatenate X and Y
    Z = torch.cat([X_flat, Y_onehot], dim=1)

    # SVD
    U, s, Vt = torch.linalg.svd(Z, full_matrices=False)

    # Lambda
    if lam is None:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(X.device)
    if not torch.is_tensor(lam):
        lam = torch.tensor(lam).to(X.device)

    # Scale singular values (simplified: scaling small singular values)
    cumperc = torch.cumsum(s, dim=0) / torch.sum(s)
    condition = cumperc > rho if small_singular else cumperc < rho
    lam_mult = torch.where(condition, lam, torch.tensor(1.0, device=s.device))
    s_scaled = s * lam_mult

    # Reconstruct Z
    Z_scaled = (U @ torch.diag(s_scaled) @ Vt)

    # Split back to X and Y
    X_flat_scaled = Z_scaled[:, :X_flat.shape[1]]
    Y_onehot_scaled = Z_scaled[:, X_flat.shape[1]:]

    # Reshape X to original image shape
    X_scaled = X_flat_scaled.view_as(X)

    # Optionally: Convert one-hot back to class labels (argmax)
    Y_scaled = torch.argmax(Y_onehot_scaled, dim=1)

    return X_scaled, Y_scaled


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

    X_scaled, Y_scaled = foma( x_dummy, y_dummy)
    print(X_scaled.shape, Y_scaled.shape)



    # # モデルのインスタンス化
    # model = SimpleCNN(feature_dim=feature_dim)

    # # ステップ1の実行
    # A = step1(model, x_dummy, y_dummy, num_classes)

    # # ステップ2の実行
    # lam = torch.Tensor([0.5]) 
    # k = 20
    # A = step2(A, k, lam)

    # # ステップ3の実行
    # Z_l_aug, Y_soft = step3(A, feature_dim, num_classes)
    # print(Z_l_aug.shape)
    # print(Y_soft)