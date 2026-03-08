import torch
import torch.nn as nn
import torch.nn.functional as F

def local_foma_fast_with_memory(
    X: torch.Tensor,             # (B, D) 特徴ベクトル (Anchor)
    Y: torch.Tensor,             # (B,) or (B, C) ラベル (Anchor)
    memory_bank,                 # FeatureMemoryBank
    num_classes: int,
    alpha: float,
    rho: float,
    k: int = 10,
    scaleup: bool = False,
    small_singular: bool = True,
    lam: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    B, D = X.shape
    device = X.device

    if Y.ndim == 1:
        Yh = F.one_hot(Y, num_classes).float()
    else:
        Yh = Y.float()
    
    Z_anchor = torch.cat([X, Yh], dim=1)

    mem_feats, mem_labels = memory_bank.get_memory()
    
    if mem_feats is None:
        return X, Yh

    if mem_labels.ndim == 1:
        mem_Yh = F.one_hot(mem_labels, num_classes).float()
    else:
        mem_Yh = mem_labels.float()
    
    Z_memory = torch.cat([mem_feats, mem_Yh], dim=1)

    Z_support = torch.cat([Z_anchor.detach(), Z_memory], dim=0)
    
    X_support = Z_support[:, :D]

    dist = torch.cdist(X, X_support)
    
    search_k = min(k - 1, Z_support.shape[0])
    if search_k < 1:
         return X, Yh

    _, nbr_indices = torch.topk(dist, k=search_k, largest=False, sorted=True) # (B, k-1)

    Z_neighbors = Z_support[nbr_indices]

    Z_batch = torch.cat([Z_anchor.unsqueeze(1), Z_neighbors], dim=1)

    Z_mean = Z_batch.mean(dim=1, keepdim=True)
    Z_centered = Z_batch - Z_mean

    try:
        U, S, Vt = torch.linalg.svd(Z_centered, full_matrices=False)
    except RuntimeError:
        return X, Yh

    # 累積寄与率とマスク
    S_sum = S.sum(dim=1, keepdim=True) + 1e-8
    cum_score = torch.cumsum(S, dim=1) / S_sum
    
    if small_singular:
        mask = cum_score > rho
    else:
        mask = cum_score < rho

    if lam is None:
        lam_val = torch.distributions.Beta(alpha, alpha).sample((B, 1)).to(device)
        if scaleup: lam_val += 1.0
    else:
        lam_val = lam
        if lam_val.ndim == 1: lam_val = lam_val.view(B, 1)

    scale = torch.where(mask, lam_val, torch.tensor(1.0, device=device))
    S_new = S * scale

    Z_rec_centered = (U * S_new.unsqueeze(1)) @ Vt
    Z_rec = Z_rec_centered + Z_mean
    
    Z_anchor_new = Z_rec[:, 0, :] # (B, D+C)

    # 特徴量とラベルに分離
    X_aug = Z_anchor_new[:, :D]
    Y_aug_logits = Z_anchor_new[:, D:]

    Y_aug = F.softmax(Y_aug_logits, dim=1)

    return X_aug, Y_aug

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

    normalized_labels = F.softmax(Y_onehot_scaled, dim=1)

    return X_scaled, normalized_labels

def local_foma(
    X: torch.Tensor,             # (B, D) 特徴ベクトル
    Y: torch.Tensor,             # (B,) あるいは (B, C) one-hot ラベル
    num_classes: int,
    alpha: float,
    rho: float,
    k: int = 10,
    scaleup: bool=False,
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

    dist_fill = dist.clone()
    dist_fill.fill_diagonal_(float("inf"))  # 自己を除外して k-1 近傍を探す
    nbr = torch.topk(dist_fill, k=k-1, largest=False, sorted=True).indices  # (B, k-1)
    self_idx = torch.arange(B, device=device).unsqueeze(1)                 # (B, 1)
    idx = torch.cat([self_idx, nbr], dim=1)        

    X_aug = torch.empty((B, D), device=device, dtype=X.dtype)
    Y_aug = torch.empty((B, num_classes), device=device, dtype=Yh.dtype)
    one = torch.tensor(1.0, device=device)

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

        if lam is None:
            lam_i = torch.distributions.Beta(alpha, alpha).sample().to(device)
            if scaleup:
                lam_i += 1
        else:
            lam_i = lam if torch.is_tensor(lam) else torch.tensor(lam, device=device)

        # 特異値縮小or拡大
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

def local_foma_fast(
    X: torch.Tensor,             # (B, D) 特徴ベクトル
    Y: torch.Tensor,             # (B,) あるいは (B, C) one-hot ラベル
    num_classes: int,
    alpha: float,
    rho: float,
    k: int = 10,
    scaleup: bool = False,
    small_singular: bool = True,
    lam: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    local_foma の完全ベクトル化による高速化実装
    (論理的な挙動は元の実装と同一)
    """
    B, D = X.shape
    device = X.device

    # 1. ラベルのOne-hot化と結合データの準備
    if Y.ndim == 1:
        Yh = F.one_hot(Y, num_classes).float()
    else:
        Yh = Y.float()
    
    # 特徴量とラベルを結合: Global Z (B, D+C)
    Z_global = torch.cat([X, Yh], dim=1)
    dim_combined = D + num_classes

    # 2. k-近傍探索 (Batch処理)
    dist = torch.cdist(X, X)
    
    dist.fill_diagonal_(float("inf"))

    _, nbr_indices = torch.topk(dist, k=k-1, largest=False, sorted=True)
    
    self_indices = torch.arange(B, device=device).view(B, 1)
    
    knn_indices = torch.cat([self_indices, nbr_indices], dim=1)

    Z_batch = Z_global[knn_indices]

    Z_mean = Z_batch.mean(dim=1, keepdim=True)
    Z_centered = Z_batch - Z_mean

    try:
        U, S, Vt = torch.linalg.svd(Z_centered, full_matrices=False)
    except RuntimeError:
        return X, F.softmax(Yh, dim=1)

    S_sum = S.sum(dim=1, keepdim=True) + 1e-8
    cum_score = torch.cumsum(S, dim=1) / S_sum
    
    if small_singular:
        mask = cum_score > rho
    else:
        mask = cum_score < rho

    if lam is None:
        # (B, 1)
        lam_val = torch.distributions.Beta(alpha, alpha).sample((B, 1)).to(device)
        if scaleup:
            lam_val += 1.0
    else:
        lam_val = lam if torch.is_tensor(lam) else torch.tensor(lam, device=device)
        if lam_val.ndim == 0:
            lam_val = lam_val.view(1, 1)
        elif lam_val.ndim == 1:
            lam_val = lam_val.view(B, 1)

    scale = torch.where(mask, lam_val, torch.tensor(1.0, device=device))
    
    S_new = S * scale

    Z_rec_centered = (U * S_new.unsqueeze(1)) @ Vt
    
    Z_rec = Z_rec_centered + Z_mean

    Z_anchor_new = Z_rec[:, 0, :] # (B, D+C)

    X_aug = Z_anchor_new[:, :D]
    Y_aug_logits = Z_anchor_new[:, D:]

    # ラベルのSoftmax化
    Y_aug = F.softmax(Y_aug_logits, dim=1)

    return X_aug, Y_aug


def compute_foma_loss(model, images, labels, k, num_classes, lambda_almp=1.0, device='cuda', scaleup=True):
    model.train()
    images = images.to(device)
    labels = labels.to(device)

    # 特徴抽出
    features = model.extract_features(images)  # (B, D)

    # 元の分類出力
    logits_orig = model.linear(features)
    loss_orig = F.cross_entropy(logits_orig, labels)

    # FOMAによる特徴摂動 
    features_foma, labels_foma = local_foma(features, labels, num_classes=num_classes, alpha=1.0, rho=0.9, k=k, scaleup=scaleup)
    logits_foma = model.linear(features_foma)
    loss_foma = F.cross_entropy(logits_foma, labels_foma)

    return loss_orig + lambda_almp * loss_foma, logits_orig
