import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# def local_pca_perturbation(data, device, k=10, alpha=1.0):
#     """
#     局所PCAに基づく摂動をデータに加える（近傍の散らばり内に収める）
#     :param data: (N, D) 次元のテンソル (N: サンプル数, D: 特徴次元)
#     :param device: 使用するデバイス（cuda or cpu）
#     :param k: k近傍の数
#     :param alpha: 摂動の強さ（最大主成分の標準偏差に対する割合）
#     :return: 摂動後のテンソル（同shape）
#     """
#     data_np = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
#     # N = data_np.shape[0]
#     # data_flat = data_np.reshape(N, -1)   # -> (N, C*H*W)
#     # # ここで
#     # N, D = data_flat.shape
#     N, D = data_np.shape
#     if N < k:
#         k = N

#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_np)
#     _, indices = nbrs.kneighbors(data_np)

#     perturbed_data = np.copy(data_np)

#     for i in range(N):
#         neighbors = data_np[indices[i]]
#         pca = PCA(n_components=min(D, k))
#         pca.fit(neighbors)
#         components = pca.components_           # shape: (n_components, D)
#         variances = pca.explained_variance_    # shape: (n_components,)

#         # ノイズベクトル（各主成分方向に沿った合成）
#         noise = np.zeros(D)
#         for j in range(len(components)):
#                 noise += np.random.randn() * np.sqrt(variances[j]) * components[j]

#         # ノイズの方向はそのまま、長さをスケールする
#         if np.linalg.norm(noise) > 0:
#             noise = noise / np.linalg.norm(noise)

#         # 局所の最大主成分の標準偏差に比例したスケール
#         max_std = np.sqrt(variances[0])  # 最大分散方向
#         scaled_noise = alpha * max_std * noise

#         scaled_noise = alpha * noise

#         perturbed_data[i] += scaled_noise

#     return torch.tensor(perturbed_data, dtype=torch.float32).to(device)


import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

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
