import torch
import torch.nn.functional as F
import numpy as np

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

def compute_hybrid_loss(model, images, labels, alpha_mix=1.0, alpha_foma=1.0, rho=0.9, k=10, gamma_mix=1.0, gamma_foma=1.0, device='cuda'):
    B = images.size(0)
    # 1) 元損失
    logits_orig = model.linear(model.extract_features(images))
    loss_orig = F.cross_entropy(logits_orig, labels)

    # 2) 入力空間 Mixup
    x_mix, y_a, y_b, lam1 = mixup_data(images, labels, alpha=alpha_mix)
    feat_mix = model.extract_features(x_mix)
    logits_mix = model.linear(feat_mix)
    loss_mix = lam1 * F.cross_entropy(logits_mix, y_a) + (1-lam1) * F.cross_entropy(logits_mix, y_b)

    # 3) Local-FOMA（ソフトラベル）
    #    y_mix_soft: (B, C)
    y_a_oh = F.one_hot(y_a, 100).float().to(device)
    y_b_oh = F.one_hot(y_b, 100).float().to(device)
    y_mix_soft = lam1 * y_a_oh + (1 - lam1) * y_b_oh

    feat_foma, _ = local_foma(feat_mix, y_mix_soft, num_classes=100, alpha=alpha_foma, rho=rho, k=k)
    logits_foma = model.linear(feat_foma)
    # ソフトラベル用クロスエントロピー
    loss_foma = -(y_mix_soft * F.log_softmax(logits_foma, dim=-1)).sum(dim=1).mean()

    # 4) 合成
    total_loss = (loss_orig + gamma_mix * loss_mix + gamma_foma * loss_foma)
    return total_loss, logits_orig

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam