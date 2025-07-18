import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def local_pca_perturbation_torch(
    features: torch.Tensor,
    k: int = 10,
    alpha: float = 0.5,
    perturb_prob: float = 1.0,
    method: str = "svd",
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
        cov = cov + torch.eye(D, device=device).unsqueeze(0) * 1e-3  # 安定化
        # 5) チョレスキー分解＋ノイズ生成
        L = torch.linalg.cholesky(cov)                  # (B, D, D)
        eps = torch.randn(B, D, device=device)          # (B, D)
        delta = alpha * (L @ eps.unsqueeze(-1)).squeeze(-1)  # (B, D)

    # 6) perturb_prob でマスクして合成 mask[i]=1のサンプルには摂動あり，=0は摂動なし
    mask = (torch.rand(B, device=device) < perturb_prob).float().unsqueeze(-1)
    features_pert = features + mask * delta

    return features_pert


def compute_almp_loss_wrn(model, images, labels, method, lambda_almp=0.5, device='cuda'):
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
    features_almp = local_pca_perturbation_torch(features, method=method)
    logits_almp = model.linear(features_almp)
    loss_almp = F.cross_entropy(logits_almp, labels)

    return loss_orig + lambda_almp * loss_almp, logits_orig


### ------------------------------------------------------------------------------------------- ###


def local_pca_perturbation(data, device, k=10, alpha=1.0):
    """
    局所PCAに基づく摂動をデータに加える（近傍の散らばり内に収める）。
    任意次元のテンソルを受け取り、最初の軸をサンプル数 N、残りを flatten して特徴次元 D として扱う。

    :param data: torch.Tensor of shape (N, ...), 任意次元
    :param device: 使用するデバイス（'cuda' or 'cpu'）
    :param k: k近傍の数
    :param alpha: 摂動の強さ（最大主成分の標準偏差に対する割合）
    :return: torch.Tensor, 摂動後のデータ（元と同shape, dtype）
    """
    # 1) Tensor→NumPy
    x_np = data.detach().cpu().numpy()
    orig_shape = x_np.shape           # e.g. (N, C, H, W) or (N, D)
    N = orig_shape[0]

    # 2) flatten to (N, D_flat)
    D_flat = int(np.prod(orig_shape[1:]))
    x_flat = x_np.reshape(N, D_flat)  # now shape = (N, D_flat)

    # 3) k > N 対策
    if k > N:
        k = N

    # 4) k-NN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs.fit(x_flat)
    _, indices = nbrs.kneighbors(x_flat)

    # 5) PCA-based noise
    pert_flat = x_flat.copy()
    for i in range(N):
        neigh = x_flat[indices[i]]              # shape (k, D_flat)
        pca = PCA(n_components=min(k, D_flat))
        pca.fit(neigh)
        comps = pca.components_                 # (n_comp, D_flat)
        vars_ = pca.explained_variance_         # (n_comp,)

        # 主成分方向に沿ったノイズベクトル生成
        noise = np.zeros(D_flat, dtype=np.float32)
        for comp_vec, var in zip(comps, vars_):
            noise += np.random.randn() * np.sqrt(var) * comp_vec

        # 正規化してスケール
        norm = np.linalg.norm(noise)
        if norm > 0:
            noise = noise / norm * (alpha * np.sqrt(vars_[0]))

        pert_flat[i] += noise

    # 6) 元の形にリシェイプ
    perturbed = pert_flat.reshape(orig_shape)

    # 7) Tensorに戻してデバイスへ
    return torch.tensor(perturbed, dtype=data.dtype, device=device)



def fast_batch_pca_perturbation(x: torch.Tensor, device: str='cuda', k: int=10, alpha: float=1.0):
    """
    バッチ単位で PCA を計算し、上位 k 成分方向にノイズを加える高速版。

    :param x: Tensor of shape (N, D) または (N, C, H, W)
    :param device: 'cuda' or 'cpu'
    :param k: 上位何成分を使うか
    :param alpha: ノイズ強度スケール
    :return: Tensor same shape as x, dtype/device も同じ
    """
    # flatten  (N, C,H,W) -> (N, D_flat)
    orig_shape = x.shape
    N = orig_shape[0]
    x_flat = x.view(N, -1).to(device)            # (N, D_flat)
    
    # 平均を引いてセンタリング
    mean = x_flat.mean(dim=0, keepdim=True)      # (1, D_flat)
    x_centered = x_flat - mean                   # (N, D_flat)
    
    # 上位 k 成分を高速に計算 (GPU 上での SVD)
    # torch.pca_lowrank は内部で SVD を使い、mean 引きも不要にしてくれるが
    # 明示的に mean を取った後のほうが挙動が分かりやすい。
    U, S, V = torch.pca_lowrank(x_centered, q=min(k, x_centered.shape[1]))
    # V: (D_flat, k), S: (k,)
    
    # 分散は S**2/(N-1) だが、スケーリングには S を直接使っても OK
    # ノイズを各主成分方向に合成
    # z ~ N(0,1) を k 個
    z = torch.randn(k, device=device)
    # noise_flat: (D_flat,)
    noise_flat = (V * (S * z)).sum(dim=1)       # 各成分 S[i]*z[i] * V[:,i]
    
    # 正規化 & α×(最大成分の標準偏差) でスケール
    norm = noise_flat.norm(p=2)
    if norm > 0:
        max_std = S[0] / ((N-1)**0.5)            # 第1主成分の std = S[0]/√(N−1)
        noise_flat = noise_flat / norm * (alpha * max_std)
    
    # バッチ全体に同じノイズを足すなら
    pert_flat = x_flat + noise_flat.unsqueeze(0)
    # もし各サンプル別々にしたいなら z を (N, k) にしてループ orバッチ化する
    
    # 元形状に戻す
    pert = pert_flat.view(orig_shape).to(x.dtype)
    return pert

# 使い方例
# x: torch.Tensor (N,C,H,W) or (N,D)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# x = x.to(device)
# x_pert = fast_batch_pca_perturbation(x, device=device, k=20, alpha=0.5)
