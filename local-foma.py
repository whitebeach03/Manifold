# Single-target Local-FOMA: augment ONLY one chosen sample and plot it with the original cloud.
# (B=50, k=10, rho=0.9). Uses matplotlib only; one figure.

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def local_foma_single(
    X: torch.Tensor,             # (B, D)
    Y: torch.Tensor,             # (B,) or (B,C) one-hot
    num_classes: int,
    alpha: float,
    rho: float,
    k: int = 10,
    scaleup: bool=False,
    small_singular: bool = True,
    lam: torch.Tensor = None,
    i_target: int = 0,
):
    """
    Local-FOMA for a single target index (augment only that point).
    Returns:
      x_aug: (D,) augmented feature for i_target
      y_soft: (C,) soft label for i_target
      nbr_idx: (k,) indices (anchor first) used to build the local neighborhood
    """
    B, D = X.shape
    device = X.device

    # one-hot 化
    if Y.ndim == 1:
      Yh = F.one_hot(Y, num_classes).float()
    else:
      Yh = Y.float()

    # kNN indices for the target
    dist = torch.cdist(X, X)                              # (B, B)
    dist_fill = dist.clone()
    dist_fill.fill_diagonal_(float("inf"))
    # nearest k-1 neighbors to i_target
    nbr = torch.topk(dist_fill[i_target], k=k-1, largest=False, sorted=True).indices  # (k-1,)
    idx = torch.cat([torch.tensor([i_target], device=device), nbr], dim=0)            # (k,)

    # Build local joint matrix
    Xi = X[idx]                                           # (k, D)
    Yi = Yh[idx]                                          # (k, C)
    Zi = torch.cat([Xi, Yi], dim=1)                       # (k, D+C)

    A  = torch.cat([Xi, Yi], dim=1)                 # (k, D+C)
    mu = A.mean(dim=0, keepdim=True)                # 列平均を保持
    Zi = A - mu                                     # 中心化

    # SVD
    U, s, Vt = torch.linalg.svd(Zi, full_matrices=False)     # U:(k,k), s:(r,), Vt:(r,D+C)
    r = s.size(0)

    # Lambda scaling
    if lam is None:
        lam_i = torch.distributions.Beta(alpha, alpha).sample().to(device)
        if scaleup:
            lam_i += 1
    else:
        lam_i = lam if torch.is_tensor(lam) else torch.tensor(lam, device=device)

    # Select small (or large) singular components via cumulative ratio in s
    cum = torch.cumsum(s, dim=0) / s.sum()
    cond = cum > rho if small_singular else cum < rho
    scale = torch.where(cond, lam_i, torch.tensor(1.0, device=device))
    print(scale)
    s2 = s * scale

    pert = U @ (torch.diag(s2 - s)) @ Vt            # 変形“差分”のみ
    A2   = Zi + pert                                # 中心化空間で加算
    row  = (A2 + mu)[0]                             # 平均を足し戻して元座標へ
    x2, y2 = row[:D], row[D:]

    # Softmax on label part
    y2 = F.softmax(y2, dim=-1)

    return x2, y2, idx.detach().cpu().numpy()

# ---- Build a small 2D dataset (B=50) ----
torch.manual_seed(42)
np.random.seed(42)

B = 128
D = 2
C = 2

mean0 = np.array([0.0, 0.0])
mean1 = np.array([2.5, 1.5])
cov0 = np.array([[0.5, 0.0],[0.0, 0.3]])
cov1 = np.array([[0.4, 0.0],[0.0, 0.6]])

X0 = np.random.multivariate_normal(mean0, cov0, size=B//2)
X1 = np.random.multivariate_normal(mean1, cov1, size=B - B//2)

X_np = np.vstack([X0, X1]).astype(np.float32)
y_np = np.array([0]*(B//2) + [1]*(B - B//2), dtype=np.int64)
perm = np.random.permutation(B)
X_np = X_np[perm]
y_np = y_np[perm]

X = torch.tensor(X_np)
Y = torch.tensor(y_np)

# ---- Choose a target index and augment only that one ----
i_target = 13
x_aug_t, y_soft_t, nbr_idx = local_foma_single(
    X=X, Y=Y, num_classes=C, alpha=1.0, rho=0.9, k=32,
    scaleup=False, small_singular=True, lam=None, i_target=i_target
)
print(x_aug_t)

# X_aug_list = []
# Y_soft_list = []
# for i in range(B):
#     x_aug_i, y_soft_i, _ = local_foma_single(
#         X=X, Y=Y, num_classes=C, alpha=1.0, rho=0.9, k=32,
#         scaleup=False, small_singular=True, lam=None, i_target=i
#     )
#     X_aug_list.append(x_aug_i.detach().cpu().numpy())
#     Y_soft_list.append(y_soft_i.detach().cpu().numpy())

# X_aug_np = np.stack(X_aug_list, axis=0)
# Y_soft_np = np.stack(Y_soft_list, axis=0)

# # ---- Plot: original cloud and ALL augmented points ----
# plt.figure()
# plt.scatter(X_np[:,0], X_np[:,1], alpha=0.35, label="Original cloud", marker=".")
# plt.scatter(X_aug_np[:,0], X_aug_np[:,1], alpha=0.8, label="Augmented (all)", marker="x")
# plt.title("Local-FOMA: all points augmented")
# plt.xlabel("x1"); plt.ylabel("x2"); plt.axis("equal"); plt.legend()
# plt.show()

x_aug_t = x_aug_t.detach().cpu().numpy()
x_orig_t = X_np[i_target]

# ---- Plot: original cloud + target before/after + its local neighborhood ----
plt.figure()
# Original cloud
plt.scatter(X_np[:,0], X_np[:,1], alpha=0.35, label="Original cloud", marker=".")
# Neighborhood points (excluding anchor) for visual context
nbr_only = [j for j in nbr_idx if j != i_target]
plt.scatter(X_np[nbr_only,0], X_np[nbr_only,1], alpha=0.8, label="k-1 neighbors", marker="s")
# Target before/after
plt.scatter([x_orig_t[0]], [x_orig_t[1]], s=120, label="Target (before)", marker="o")
plt.scatter([x_aug_t[0]], [x_aug_t[1]], s=120, label="Target (after)", marker="x")
# Arrow
plt.arrow(x_orig_t[0], x_orig_t[1], x_aug_t[0]-x_orig_t[0], x_aug_t[1]-x_orig_t[1],
        length_includes_head=True, head_width=0.05, head_length=0.1, alpha=0.9)
plt.title(f"Local-FOMA (single target): index={i_target}, k=10, rho=0.9")

# --- k近傍集合のPCAベクトル描画 ---
local_pts = X_np[nbr_idx]                   # アンカー＋k-1近傍
local_mean = local_pts.mean(axis=0)
X_loc_centered = local_pts - local_mean
_, S_loc, Vt_loc = np.linalg.svd(X_loc_centered, full_matrices=False)
PC1_loc, PC2_loc = Vt_loc[0], Vt_loc[1]

scale = 2.0  # 可視化スケール（必要なら S_loc に比例させてもよい）

plt.arrow(local_mean[0], local_mean[1], PC1_loc[0]*scale, PC1_loc[1]*scale,
        head_width=0.1, head_length=0.2, color="red", alpha=0.8)
plt.arrow(local_mean[0], local_mean[1], PC2_loc[0]*scale, PC2_loc[1]*scale,
        head_width=0.1, head_length=0.2, color="green", alpha=0.8)

plt.text(local_mean[0] + PC1_loc[0]*scale*1.1, local_mean[1] + PC1_loc[1]*scale*1.1, "local PC1", color="red")
plt.text(local_mean[0] + PC2_loc[0]*scale*1.1, local_mean[1] + PC2_loc[1]*scale*1.1, "local PC2", color="green")


# X_centered = X_np - X_np.mean(axis=0)
# U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
# PC1, PC2 = Vt[0], Vt[1]
# mean = X_np.mean(axis=0)
# scale = 2.0  # 可視化スケール

# plt.arrow(mean[0], mean[1], PC1[0]*scale, PC1[1]*scale,
#           head_width=0.1, head_length=0.2, color="red", alpha=0.8)
# plt.arrow(mean[0], mean[1], PC2[0]*scale, PC2[1]*scale,
#           head_width=0.1, head_length=0.2, color="green", alpha=0.8)

# plt.text(mean[0] + PC1[0]*scale*1.1, mean[1] + PC1[1]*scale*1.1, "PC1", color="red")
# plt.text(mean[0] + PC2[0]*scale*1.1, mean[1] + PC2[1]*scale*1.1, "PC2", color="green")

plt.xlabel("x1"); plt.ylabel("x2"); plt.axis("equal"); plt.legend()
plt.show()
plt.savefig(f"./local_foma/{i_target}.png", bbox_inches="tight")

    # Also return useful values for reference
(i_target, x_orig_t, x_aug_t, y_soft_t.detach().cpu().numpy(), nbr_idx)
