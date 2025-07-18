import torch
import torch.nn.functional as F

def local_svd_perturbation_torch(
    features: torch.Tensor,
    k: int = 10,
    alpha: float = 0.5,
    perturb_prob: float = 1.0,
    energy: float = 0.9
) -> torch.Tensor:
    """
    SVDベースの局所PCA摂動（累積寄与率に応じて各サンプルごとにmを決定）

    Args:
        features: (B, D) バッチ中間特徴ベクトル
        k: 近傍数
        alpha: 摂動強度
        perturb_prob: 摂動を適用する確率
        energy: 累積寄与率の閾値 (0<energy<=1)

    Returns:
        features_pert: (B, D) 摂動後特徴
    """
    B, D = features.shape
    device = features.device

    # 1) k-NN idx
    dist = torch.cdist(features, features)                         # (B, B)
    dist += torch.eye(B, device=device) * 1e6                      # 自己距離除外
    idx = torch.topk(dist, k=k, largest=False).indices             # (B, k)

    # 2) neighbors & 中心化 → (B, k, D)
    neighbors = features[idx]
    mu        = neighbors.mean(dim=1, keepdim=True)
    centered  = neighbors - mu

    # 3) SVD: U:(B,k,k), S:(B,r), Vh:(B,r,D) where r=min(k,D)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    r        = S.size(1)

    # 4) 各サンプルごとに m_i を決定しノイズ生成
    deltas = []
    for i in range(B):
        # 各成分の寄与率
        s2 = S[i]**2                                            # (r,)
        ratios = s2 / s2.sum()                                  # (r,)
        cumrat = torch.cumsum(ratios, dim=0)                    # (r,)
        # 最小の m_i を見つける
        m_i = int((cumrat >= energy).nonzero()[0]) + 1          # 1 以上の整数

        # 主成分行列と固有値 (B, D, m_i),(m_i,)
        V_top = Vh[i, :m_i, :].transpose(0,1)                   # (D, m_i)
        lambda_sqrt = torch.sqrt(s2[:m_i] / (k - 1))            # (m_i,)

        # ノイズ eps
        eps = torch.randn(m_i, device=device)                   # (m_i,)
        delta_i = alpha * (V_top @ (lambda_sqrt * eps))         # (D,)
        deltas.append(delta_i)

    delta = torch.stack(deltas, dim=0)                          # (B, D)

    # 5) mask で確率的に適用
    mask = (torch.rand(B, device=device) < perturb_prob).float().unsqueeze(-1)  # (B,1)
    return features + mask * delta

def compute_almp_loss_svd(model, images, labels, lambda_almp=1.0, device='cuda'):
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
    features_almp = local_svd_perturbation_torch(features)
    logits_almp = model.linear(features_almp)
    loss_almp = F.cross_entropy(logits_almp, labels)

    return loss_orig + lambda_almp * loss_almp, logits_orig