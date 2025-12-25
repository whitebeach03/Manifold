import torch
import torch.nn as nn
import torch.nn.functional as F

def unrestricted_foma(
    x_anchor: torch.Tensor, 
    y_anchor: torch.Tensor, 
    k: int = 4, 
    alpha: float = 1.0, 
    rho: float = 0.9,
    num_classes: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unrestricted Local-FOMA (Batch Only)
    ミニバッチ内のデータのみを使用し、クラス制約なしで近傍探索を行う。
    特徴量とラベルの双方をSVDを用いて多様体上で変形する。
    
    Args:
        x_anchor (Tensor): 入力バッチの特徴量 (B, D)
        y_anchor (Tensor): 入力バッチのラベル (B,) or (B, C)
        k (int): 近傍数 (自身を含むため、最低でも2以上推奨)
        alpha (float): Beta分布のパラメータ
        rho (float): 支配的成分の累積寄与率閾値
        num_classes (int): クラス数 (Noneの場合はバッチ内の最大値から推論)
        
    Returns:
        x_aug (Tensor): 摂動後の特徴量 (B, D)
        y_aug (Tensor): SVD変形されたソフトラベル (B, C)
    """
    B, D = x_anchor.shape
    device = x_anchor.device
    
    # --- 前処理: ラベル処理 ---
    # One-hotならIndexに戻す（近傍探索の便宜上）
    if y_anchor.ndim > 1:
        y_idx = y_anchor.argmax(dim=1)
    else:
        y_idx = y_anchor
        
    # クラス数の確定
    if num_classes is None:
        num_classes = y_idx.max().item() + 1

    # --- 1. バッチ内 近傍探索 (クラス制約なし) ---
    # 自分自身を含む距離行列 (B, B)
    dist = torch.cdist(x_anchor, x_anchor)
    
    # バッチサイズがkより小さい場合のガード
    search_k = min(k, B)
    
    # 距離が近い順にインデックスを取得
    # 通常、自分自身(距離0)が最上位に来る
    knn_vals, knn_indices = torch.topk(dist, k=search_k, largest=False, sorted=True) # (B, k)
    
    # --- 2. 局所多様体構成 (Features & Labels) ---
    # 近傍の特徴量を取得 (B, k, D)
    Z_i = x_anchor[knn_indices]
    
    # 近傍のラベルを取得してOne-Hot化 (B, k, C)
    neighbor_labels = y_idx[knn_indices]
    L_i = F.one_hot(neighbor_labels, num_classes=num_classes).float()
    
    # ※ 数値誤差で自分自身がindex 0に来ない可能性を排除するため、
    # 明示的にindex 0をAnchor自身で上書きする (FOMAの定義: Anchor中心)
    Z_i[:, 0, :] = x_anchor
    L_i[:, 0, :] = F.one_hot(y_idx, num_classes=num_classes).float()

    # --- 3. SVD & Perturbation ---
    # 特徴量の中心化
    Z_mean = Z_i.mean(dim=1, keepdim=True)
    Z_centered = Z_i - Z_mean
    
    # ラベルの中心化
    L_mean = L_i.mean(dim=1, keepdim=True)
    L_centered = L_i - L_mean

    # 特徴量に微小ノイズ付加 (SVD安定化)
    jitter = 1e-4  
    Z_centered_noise = Z_centered + torch.randn_like(Z_centered) * jitter
    
    # SVD実行: Z = U S V^T
    # U: (B, k, k), S: (B, k)
    try:
        U, S, Vt = torch.linalg.svd(Z_centered_noise, full_matrices=False)
    except RuntimeError:
        # SVD失敗時は元のデータをそのまま返す（One-hot化して）
        return x_anchor, F.one_hot(y_idx, num_classes=num_classes).float()

    # 累積寄与率とスケーリング係数の計算
    S_sum = S.sum(dim=1, keepdim=True) + 1e-8
    cum_score = torch.cumsum(S, dim=1) / S_sum
    mask_dominant = cum_score <= rho
    
    # 摂動係数 lambda ~ Beta(alpha, alpha)
    lam = torch.distributions.Beta(alpha, alpha).sample((B, 1)).to(device)
    
    # 主要成分は維持(1.0倍)、ノイズ成分は縮小(lambda倍)
    scale = torch.where(mask_dominant, torch.tensor(1.0, device=device), lam) # (B, k)
    
    # --- 4. 特徴量の再構成 ---
    # Z_new = Mean + U * (S * scale) * Vt
    S_scaled = S * scale
    rec_feats = U @ (S_scaled.unsqueeze(2) * Vt)
    Z_new = rec_feats + Z_mean
    
    x_aug = Z_new[:, 0, :] # 中心(Anchor)に対応するデータを取り出す
    
    # --- 5. ラベルの再構成 (Soft Label) ---
    # ラベルも特徴空間の構造(U)に従って変形する
    # L_new = Mean + U * (Projected_Coeffs * scale)
    # Projected_Coeffs = U^T @ L_centered
    
    # U^T: (B, k, k) @ L_centered: (B, k, C) -> 係数: (B, k, C)
    L_coeffs = torch.bmm(U.transpose(1, 2), L_centered)
    
    # 係数をスケーリング (scaleを(B,k,1)に拡張)
    L_coeffs_scaled = L_coeffs * scale.unsqueeze(2)
    
    # 再構成: U @ coeffs_scaled + Mean
    L_rec = torch.bmm(U, L_coeffs_scaled) + L_mean
    
    y_aug_raw = L_rec[:, 0, :] # Anchorに対応するソフトラベル
    
    # --- 6. ラベルの正規化 ---
    # 値が [0, 1] を超えたり負になるのを防ぎ、合計を1にする
    y_aug = torch.clamp(y_aug_raw, min=0.0)
    y_sum = y_aug.sum(dim=1, keepdim=True) + 1e-8
    y_aug = y_aug / y_sum
    
    return x_aug, y_aug

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
    # dist += torch.eye(B, device=device) * 1e6
    # idx  = torch.topk(dist, k=k, largest=False).indices           # (B, k)
    dist_fill = dist.clone()
    dist_fill.fill_diagonal_(float("inf"))  # 自己を除外して k-1 近傍を探す
    nbr = torch.topk(dist_fill, k=k-1, largest=False, sorted=True).indices  # (B, k-1)
    self_idx = torch.arange(B, device=device).unsqueeze(1)                 # (B, 1)
    idx = torch.cat([self_idx, nbr], dim=1)        

    # --- ここから追加 ---
    # knn_dists = dist.gather(1, idx)                       # (B, k)
    # avg_knn_dist = knn_dists.mean().item()                    # scalar
    # print(f"Avg {k}-NN distance: {avg_knn_dist:.4f}")
    # --- ここまで追加 ---

    # X_aug = torch.empty_like(X)
    # Y_aug = torch.empty_like(Yh)
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

        # λ の用意（サンプル共通 or per-sample）
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
