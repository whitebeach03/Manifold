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
    """
    local_foma_fast に メモリバンク を導入した実装。
    特徴量とラベルを結合([X, Y])してSVDを行い、メモリバンク内の近傍(他クラス含む)を利用して
    多様体を構成し、特徴量とソフトラベルを同時に生成する。
    """
    B, D = X.shape
    device = X.device

    # --- 1. Anchorの結合データ(Z_anchor)作成 ---
    if Y.ndim == 1:
        Yh = F.one_hot(Y, num_classes).float()
    else:
        Yh = Y.float()
    
    # Anchor: (B, D+C)
    Z_anchor = torch.cat([X, Yh], dim=1)

    # --- 2. Memory Bank からのデータ取得とSupport Set構築 ---
    mem_feats, mem_labels = memory_bank.get_memory()
    
    if mem_feats is None:
        # メモリバンクが空なら、通常のバッチ内処理(local_foma_fast)にフォールバック
        # または恒等写像を返す
        return X, Yh

    # Memory Labels を One-hot 化
    if mem_labels.ndim == 1:
        mem_Yh = F.one_hot(mem_labels, num_classes).float()
    else:
        mem_Yh = mem_labels.float()
    
    # Memory: (M, D+C)
    Z_memory = torch.cat([mem_feats, mem_Yh], dim=1)

    # Support Set (Anchor(勾配切る) + Memory): (B+M, D+C)
    # これが近傍探索のプールになる
    Z_support = torch.cat([Z_anchor.detach(), Z_memory], dim=0)
    
    # 距離計算用の特徴量部分のみ抽出: (B+M, D)
    X_support = Z_support[:, :D]

    # --- 3. 近傍探索 (Unrestricted KNN) ---
    # Anchor(X) と Support(X_support) の距離行列 (B, B+M)
    # ※ ラベル情報(Y)は距離計算には含めない（見た目が似ているものを探すため）
    dist = torch.cdist(X, X_support)
    
    # 自分自身(index 0~B-1)を除外したい場合はここでinf埋めをするが、
    # FOMAは「自分+近傍」で構成するため、自分を含めてtop-kをとるのが自然。
    # ただし、確実に自分をindex 0に置くために、k-1個を探索してあとで結合する。
    
    # 自分自身(対角ブロック付近)を無限大にして、純粋な近傍だけをk-1個探す
    # dist[:, :B].fill_diagonal_(float("inf")) # 必要に応じて
    
    # k-1個の近傍を取得
    search_k = min(k - 1, Z_support.shape[0])
    if search_k < 1:
         return X, Yh

    _, nbr_indices = torch.topk(dist, k=search_k, largest=False, sorted=True) # (B, k-1)

    # --- 4. 局所バッチ(Z_batch)の構築 ---
    # 近傍データをSupport SetからGather: (B, k-1, D+C)
    Z_neighbors = Z_support[nbr_indices]
    
    # Anchor(勾配あり) を先頭(index 0)に追加して結合
    # Z_batch: (B, k, D+C)
    Z_batch = torch.cat([Z_anchor.unsqueeze(1), Z_neighbors], dim=1)

    # --- 5. Batched SVD (local_foma_fastと同一ロジック) ---
    # 中心化
    Z_mean = Z_batch.mean(dim=1, keepdim=True)
    Z_centered = Z_batch - Z_mean

    # SVD実行: Input (B, k, D+C)
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

    # Lambda (Perturbation Scale)
    if lam is None:
        lam_val = torch.distributions.Beta(alpha, alpha).sample((B, 1)).to(device)
        if scaleup: lam_val += 1.0
    else:
        lam_val = lam
        if lam_val.ndim == 1: lam_val = lam_val.view(B, 1)

    scale = torch.where(mask, lam_val, torch.tensor(1.0, device=device))
    S_new = S * scale

    # --- 6. 再構成と分割 ---
    # Z_rec = Mean + U @ S_new @ Vt
    # Broadcasting: (B, k, r) * (B, 1, r) -> (B, k, r)
    Z_rec_centered = (U * S_new.unsqueeze(1)) @ Vt
    Z_rec = Z_rec_centered + Z_mean
    
    # Anchor(index 0)を取り出し
    Z_anchor_new = Z_rec[:, 0, :] # (B, D+C)

    # 特徴量とラベルに分離
    X_aug = Z_anchor_new[:, :D]
    Y_aug_logits = Z_anchor_new[:, D:]

    # ラベルの正規化 (Softmax or Clamp&Normalize)
    # FOMA論文的にはSoftmaxだが、Mixup的にはLinear interpolateなので
    # 負値をクリップして正規化する方が分布が保存されやすい場合がある。
    # ここでは元の実装に合わせて Softmax を採用
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
    # 距離行列計算 (B, B)
    dist = torch.cdist(X, X)
    
    # 自分自身を除外するために無限大を埋める
    dist.fill_diagonal_(float("inf"))
    
    # 近傍k-1個のインデックスを取得 (B, k-1)
    # ※ バッチサイズがkより小さい場合のエラーハンドリングが必要ならmin(k-1, B-1)とする
    _, nbr_indices = torch.topk(dist, k=k-1, largest=False, sorted=True)
    
    # 自分自身のインデックス (B, 1)
    self_indices = torch.arange(B, device=device).view(B, 1)
    
    # 自分を先頭にして結合: (B, k)
    knn_indices = torch.cat([self_indices, nbr_indices], dim=1)

    # 3. 局所データの抽出 (Gather)
    # Global Z からインデックスを使ってバッチ一括抽出
    # (B, D+C) -> (B, k, D+C)
    Z_batch = Z_global[knn_indices]

    # 4. 中心化 (Batch処理)
    # 平均: (B, 1, D+C)
    Z_mean = Z_batch.mean(dim=1, keepdim=True)
    Z_centered = Z_batch - Z_mean

    # 5. Batched SVD (ここが高速化の肝)
    # 入力が (B, k, D+C) の場合、出力は
    # U: (B, k, k), S: (B, min(k, D+C)), Vt: (B, min(k, D+C), D+C)
    try:
        U, S, Vt = torch.linalg.svd(Z_centered, full_matrices=False)
    except RuntimeError:
        # SVD失敗時は元の値を返す（安全策）
        return X, F.softmax(Yh, dim=1)

    # 6. 特異値のスケーリング係数計算 (Batch処理)
    # 累積寄与率: (B, min_rank)
    S_sum = S.sum(dim=1, keepdim=True) + 1e-8
    cum_score = torch.cumsum(S, dim=1) / S_sum
    
    # マスク作成 (small_singular=Trueなら、累積寄与率が高い「末尾」の方をTrueにする)
    if small_singular:
        mask = cum_score > rho
    else:
        mask = cum_score < rho

    # Lambdaの準備
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

    # スケール適用: マスクがTrueの部分にlam_valを、それ以外は1.0を適用
    # (B, min_rank)
    scale = torch.where(mask, lam_val, torch.tensor(1.0, device=device))
    
    # 特異値の更新
    S_new = S * scale

    # 7. 再構成 (Batch MatMul)
    # U @ diag(S_new) @ Vt
    # diag行列を作る代わりにブロードキャストを利用
    # (B, k, min_rank) * (B, 1, min_rank) -> (B, k, min_rank)
    # ※ Vt は (B, min_rank, D+C)
    
    # 計算: U に S_new を掛け合わせてから Vt と積をとる
    # U: (B, k, r), S_new: (B, r) -> U * S_new.unsqueeze(1): (B, k, r)
    # (B, k, r) @ (B, r, D+C) -> (B, k, D+C)
    Z_rec_centered = (U * S_new.unsqueeze(1)) @ Vt
    
    # 平均を足す
    Z_rec = Z_rec_centered + Z_mean

    # 8. アンカー(自分自身)の取り出しと分割
    # knn_indicesの0番目が自分自身だったので、dim=1の0番目を取り出す
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
