import torch
import torch.nn.functional as F

def local_pca_perturbation_torch(
    features: torch.Tensor,
    k: int = 10,
    alpha: float = 0.5,
    perturb_prob: float = 1.0,
    method: str,
) -> torch.Tensor:
    """
    features: (B, D)
    """
    B, D = features.size()
    device = features.device

    # 1) 距離行列 → k近傍インデックス
    dist = torch.cdist(features, features)  # (B, B)
    inf_mask = torch.eye(B, device=device) * 1e6
    idx = torch.topk(dist + inf_mask, k=k, largest=False).indices  # (B, k)
    # print(idx[0])

    # 2) 近傍特徴を集める → (B, k, D)
    neighbors = features[idx]

    # 3) 中心化
    mu = neighbors.mean(dim=1, keepdim=True)    # (B, 1, D)
    centered = neighbors - mu                   # (B, k, D)

    if method == "svd":
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        eps = torch.randn(B, S.size(1), device=device) 
        sqrt_lambda = (S**2 / (k-1)).sqrt()       
        # δ = V @ (√λ · ε)
        delta = (Vh.transpose(-2,-1) @ (sqrt_lambda.unsqueeze(-1) * eps.unsqueeze(-1))).squeeze(-1) 
    elif method == "pca":
        cov = torch.matmul(centered.transpose(1, 2), centered) / (k - 1)
        cov = cov + torch.eye(D, device=device).unsqueeze(0) * 1e-5  # 安定化
        eigvals, eigvecs = torch.linalg.eigh(cov)           # (B, D), (B, D, D)
        sqrt_vals = torch.sqrt(torch.clamp(eigvals, min=0.0))  # (B, D)
        eps = torch.randn(B, D, device=device)                # (B, D)
        delta = (eigvecs @ (sqrt_vals.unsqueeze(-1) * eps.unsqueeze(-1))).squeeze(-1)  # (B, D)
    elif method == "cholesky":
        # 4) 共分散を特徴次元方向で計算 → (B, D, D)
        cov = torch.matmul(centered.transpose(1, 2), centered) / (k - 1)
        cov = cov + torch.eye(D, device=device).unsqueeze(0) * 1e-5  # 安定化
        # 5) チョレスキー分解＋ノイズ生成
        L = torch.linalg.cholesky(cov)                  # (B, D, D)
        eps = torch.randn(B, D, device=device)          # (B, D)
        delta = alpha * (L @ eps.unsqueeze(-1)).squeeze(-1)  # (B, D)

    # 6) perturb_prob でマスクして合成 mask[i]=1のサンプルには摂動あり，=0は摂動なし
    mask = (torch.rand(B, device=device) < perturb_prob).float().unsqueeze(-1)
    features_pert = features + mask * delta

    return features_pert


def compute_almp_loss_wrn(model, images, labels, lambda_almp=0.5, device='cuda'):
    model.train()
    images = images.to(device)
    labels = labels.to(device)

    # 特徴抽出
    features = model.extract_features(images)  # (B, D)

    # 元の分類出力
    logits_orig = model.linear(features)
    loss_orig = F.cross_entropy(logits_orig, labels)

    # ALMPによる特徴摂動
    # features_almp = adaptive_local_manifold_perturbation(features, device=device)
    # features_almp = local_pca_perturbation(features, device=device)
    features_almp = local_pca_perturbation_torch(features)
    logits_almp = model.linear(features_almp)
    loss_almp = F.cross_entropy(logits_almp, labels)

    return loss_orig + lambda_almp * loss_almp, logits_orig

