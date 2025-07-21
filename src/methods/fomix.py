import torch
import torch.nn.functional as F

def construct_local_Z(
    features: torch.Tensor,      # (B, D)
    Yh: torch.Tensor,            # (B, C) one-hot labels
    idx: torch.Tensor,           # indices of samples (B,)
    k: int                       # number of neighbors (including itself)
) -> torch.Tensor:
    """
    For each sample b, gather its k nearest neighbors (including itself) in feature space,
    concatenate neighbor features and one-hot labels -> (B, k, D+C)
    """
    dist = torch.cdist(features, features)                    # (B, B)
    knn  = dist.topk(k=k, largest=False).indices              # (B, k)
    neigh_idx = knn[idx]                                      # (B, k)
    Xi = features[neigh_idx]                                  # (B, k, D)
    Yi = Yh[neigh_idx]                                        # (B, k, C)
    Z = torch.cat([Xi, Yi], dim=-1)                           # (B, k, D+C)
    return Z


def fomix_pair(
    features: torch.Tensor,      # (B, D)
    labels: torch.Tensor,        # (B,) integer labels or (B, C) soft labels
    alpha: float,
    rho: float,
    k: int
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Perform Local-FOMix by mixing each feature with a shuffled partner along its local manifold.
    Returns:
        X_mix: (B, D) mixed features
        y_a:   (B,) original integer labels
        y_b:   (B,) shuffled integer labels
        lam:   (B,) mixing coefficients
    """
    B, D = features.shape
    device = features.device

    # 1) 整数ラベル & one-hot 準備
    if labels.ndim == 1:
        int_labels = labels
    else:
        int_labels = labels.argmax(dim=1)
    C = int(int_labels.max().item()) + 1
    Yh = F.one_hot(int_labels, C).float().to(device)  # (B, C)

    # 2) バッチシャッフルによるペアリング
    idx = torch.randperm(B, device=device)
    i, j = torch.arange(B, device=device), idx

    # 3) ラベルペア
    y_a = int_labels[i]   # (B,)
    y_b = int_labels[j]   # (B,)

    # 4) 局所行列の構築
    Zi_all = construct_local_Z(features, Yh, i, k)  # (B, k, D+C)
    Zj_all = construct_local_Z(features, Yh, j, k)  # (B, k, D+C)

    # 5) λ をサンプリング
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)  # (B,)

    # 6) 出力テンソル用意
    X_mix = torch.empty((B, D), device=device, dtype=features.dtype)
    Y_mix = torch.empty((B, C), device=device, dtype=Yh.dtype)

    # 7) 各サンプルごとに SVD → 特異値スケーリング → 再構成 → 補間
    for b in range(B):
        # 中心化
        Zi = Zi_all[b] - Zi_all[b].mean(dim=0)  # (k, D+C)
        Zj = Zj_all[b] - Zj_all[b].mean(dim=0)

        # SVD
        Ui, si, Vti = torch.linalg.svd(Zi, full_matrices=False)
        Uj, sj, Vtj = torch.linalg.svd(Zj, full_matrices=False)

        # --- ここから特異値スケーリング ---
        # 累積寄与率を計算
        cum_i = torch.cumsum(si, dim=0) / si.sum()
        cum_j = torch.cumsum(sj, dim=0) / sj.sum()

        # ρ を超える特異値は lam[b] 倍、それ以外はそのまま
        scale_i = torch.where(cum_i > rho, lam[b], torch.tensor(1.0, device=device))
        scale_j = torch.where(cum_j > rho, lam[b], torch.tensor(1.0, device=device))

        # 新しい特異値ベクトル
        s2_i = si * scale_i
        s2_j = sj * scale_j

        # 再構成
        Zi_rec = Ui @ torch.diag(s2_i) @ Vti  # (k, D+C)
        Zj_rec = Uj @ torch.diag(s2_j) @ Vtj
        # --- ここまで特異値スケーリング ---

        # 自身(行0)のみ抽出して特徴次元に戻す
        xi_rec = Zi_rec[0, :D]  # (D,)
        xj_rec = Zj_rec[0, :D]  # (D,)

        # 特徴とラベルの補間
        X_mix[b] = (1 - lam[b]) * xi_rec + lam[b] * xj_rec
        Y_mix[b] = (1 - lam[b]) * Yh[i[b]] + lam[b] * Yh[j[b]]

    return X_mix, Y_mix, y_a, y_b, lam


def compute_fomix_loss(
    model,                      # CNN model
    images: torch.Tensor,       # (B, C, H, W)
    labels: torch.Tensor,       # (B,) integer labels
    alpha: float = 1.0,
    rho: float = 0.9,
    k: int = 10,
    lambda_mix: float = 1.0,
    device: str = 'cuda'
) -> (torch.Tensor, torch.Tensor):
    """
    Compute loss as CE on original + lambda_mix * FOMix mixup-style loss.
    Returns total scalar loss and original logits.
    """
    model.train()
    images = images.to(device)
    labels = labels.to(device)

    # Base forward
    features = model.extract_features(images)   # (B, D)
    logits_orig = model.linear(features)        # (B, num_classes)
    loss_ce = F.cross_entropy(logits_orig, labels)

    # Generate mixed features and label pairs
    feat_mix, _, y_a, y_b, lam = fomix_pair(features, labels, alpha, rho, k)
    logits_mix = model.linear(feat_mix)

    # Mixup-style loss with per-sample coefficients and reduction to scalar
    loss_a = F.cross_entropy(logits_mix, y_a, reduction='none')  # (B,)
    loss_b = F.cross_entropy(logits_mix, y_b, reduction='none')  # (B,)
    loss_mix = ((1 - lam) * loss_a + lam * loss_b).mean()        # scalar

    total_loss = loss_ce + lambda_mix * loss_mix
    return total_loss, logits_orig
