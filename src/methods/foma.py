import torch
import torch.nn.functional as F

def foma(X, Y, num_classes, alpha, rho, small_singular=True, lam=None):
    B = X.shape[0]
    # Flatten image to [B, C*H*W]
    # X_flat = X.view(B, -1)
    X_flat = X

    # Convert labels to one-hot if needed
    if Y.ndim == 1:  # [B]
        Y_onehot = F.one_hot(Y, num_classes=num_classes).float()
    else:
        Y_onehot = Y.float()

    # Concatenate X and Y
    Z = torch.cat([X_flat, Y_onehot], dim=1)
    Z = Z - Z.mean(dim=0, keepdim=True)
    # Z = Z + 1e-3 * torch.randn_like(Z)

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
#     Y_onehot_scaled = torch.clamp(Y_onehot_scaled, min=0)
    
#     # 各サンプルごとに合計を計算
#     sum_per_sample = Y_onehot_scaled.sum(dim=1, keepdim=True)
    
#     normalized_labels = torch.where(
#     sum_per_sample == 0,
#     torch.ones_like(Y_onehot_scaled) / Y_onehot_scaled.size(1),  # 全要素0なら均等分布
#     Y_onehot_scaled / sum_per_sample
# )
    normalized_labels = F.softmax(Y_onehot_scaled, dim=1)

    return X_scaled, normalized_labels


def compute_foma_loss(model, images, labels, augment, lambda_almp=1.0, device='cuda'):
    model.train()
    images = images.to(device)
    labels = labels.to(device)

    # 特徴抽出
    features = model.extract_features(images)  # (B, D)

    # 元の分類出力
    logits_orig = model.linear(features)
    loss_orig = F.cross_entropy(logits_orig, labels)

    # FOMAによる特徴摂動
    if augment == "FOMA"
        features_foma, labels_foma = foma(features, labels, num_classes=100, alpha=1.0, rho=0.9)
    elif augment == "Local-FOMA":
        features_foma, labels_foma = local_foma(features, labels, num_classes=100, alpha=1.0, rho=0.9)
    logits_foma = model.linear(features_foma)
    loss_foma = F.cross_entropy(logits_foma, labels_foma)

    return loss_orig + lambda_almp * loss_foma, logits_orig


def local_foma(
    X: torch.Tensor,             # (B, D) 特徴ベクトル
    Y: torch.Tensor,             # (B,) あるいは (B, C) one-hot ラベル
    num_classes: int,
    alpha: float,
    rho: float,
    k: int = 10,
    small_singular: bool = True,
    lam: torch.Tensor = None,
) -> (torch.Tensor, torch.Tensor):
    """
    各サンプル i について、自身 + k-1 近傍だけを使った局所 FOMA
    """
    B, D = X.shape
    device = X.device

    # one-hot 化
    if Y.ndim == 1:
        Yh = F.one_hot(Y, num_classes).float()
    else:
        Yh = Y.float()

    # 距離行列 + 自己除外 + kNN
    dist = torch.cdist(X, X)                                      # (B, B)
    dist += torch.eye(B, device=device) * 1e6
    idx  = torch.topk(dist, k=k, largest=False).indices           # (B, k)

    X_aug = torch.empty_like(X)
    Y_aug = torch.empty_like(Yh)

    for i in range(B):
        # --- 局所 Z_i の構成 ---
        Xi = X[idx[i]]                                           # (k, D)
        Yi = Yh[idx[i]]                                         # (k, C)
        Zi = torch.cat([Xi, Yi], dim=1)                         # (k, D+C)

        # 中心化
        Zi = Zi - Zi.mean(dim=0, keepdim=True)

        # SVD
        U, s, Vt = torch.linalg.svd(Zi, full_matrices=False)     # U:(k,k), s:(r,), Vt:(r,D+C)
        r = s.size(0)

        # λ の用意（サンプル共通 or per-sample）
        if lam is None:
            lam_i = torch.distributions.Beta(alpha, alpha).sample().to(device)
        else:
            lam_i = lam if torch.is_tensor(lam) else torch.tensor(lam, device=device)

        # 特異値縮小
        cum = torch.cumsum(s, dim=0) / s.sum()
        cond = cum > rho if small_singular else cum < rho
        scale = torch.where(cond, lam_i, torch.tensor(1.0, device=device))
        s2 = s * scale

        # 再構成
        Zi2 = (U @ torch.diag(s2) @ Vt)                          # (k, D+C)

        # i番目（自身）の行を取り出し
        row = Zi2[0]                                            # (D+C,)
        x2, y2 = row[:D], row[D:]

        # ラベルはソフトマックス化
        y2 = F.softmax(y2, dim=-1)

        X_aug[i] = x2
        Y_aug[i] = y2

    return X_aug, Y_aug