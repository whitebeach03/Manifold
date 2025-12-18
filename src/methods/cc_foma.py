import torch
import torch.nn.functional as F

def cc_foma(
    x_anchor: torch.Tensor, 
    y_anchor: torch.Tensor, 
    memory_bank, 
    k: int = 10, 
    alpha: float = 1.0, 
    rho: float = 0.9
) -> torch.Tensor:
    """
    Class-Conditional Local-FOMA (CC-FOMA) Implementation
    
    Args:
        x_anchor (Tensor): 入力バッチの特徴量 (B, D)
        y_anchor (Tensor): 入力バッチのラベル (B,) or (B, C)
        memory_bank: FeatureMemoryBank インスタンス
        k (int): 近傍数 (自身を含む)
        alpha (float): Beta分布のパラメータ Beta(alpha, alpha)
        rho (float): 支配的成分の累積寄与率閾値
        
    Returns:
        x_aug (Tensor): 摂動後の特徴量 (B, D)
    """
    B, D = x_anchor.shape
    device = x_anchor.device
    
    # ラベルの整形 (Index形式に統一)
    if y_anchor.ndim > 1:
        y_idx = y_anchor.argmax(dim=1)
    else:
        y_idx = y_anchor
        
    # メモリバンクから参照用データを取得
    mem_feats, mem_labels = memory_bank.get_memory()
    
    # メモリバンクが空の場合(学習初期)は恒等写像を返す
    if mem_feats is None:
        return x_anchor

    # --- 1. Support Set (探索対象) の構築 ---
    # 現在のバッチ(勾配なし) + メモリバンク
    support_feats = torch.cat([x_anchor.detach(), mem_feats], dim=0) # (B+M, D)
    support_labels = torch.cat([y_idx, mem_labels], dim=0)           # (B+M,)
    
    # --- 2. クラス条件付き距離計算 ---
    # x_anchor と support_feats の距離行列 (B, B+M)
    # ※ D次元が大きい場合、cosine距離の方が良い場合もあるが、FOMAの定義(ユークリッド)に従う
    dist = torch.cdist(x_anchor, support_feats)
    
    # マスク作成: 同じクラスのみ True
    # (B, 1) == (1, B+M) -> (B, B+M)
    class_mask = (y_idx.unsqueeze(1) == support_labels.unsqueeze(0))
    
    # 異なるクラスの距離を無限大にする
    dist.masked_fill_(~class_mask, float("inf"))
    
    # 自分自身との距離(対角ブロック)を無限大にして、近傍として自分を選ばないようにする
    # (ただし、今回は近傍集合に自分を明示的に後で足すので、探索からは除外する)
    dist[:, :B].fill_diagonal_(float("inf")) # バッチ内自己参照を除外
    
    # --- 3. 近傍探索 ---
    # 有効な近傍を探す (k-1個)
    search_k = k - 1
    # サポートセットサイズが小さい場合のガード
    search_k = min(search_k, support_feats.shape[0])
    
    if search_k < 1:
        return x_anchor

    # 小さい順に取得
    knn_vals, knn_indices = torch.topk(dist, k=search_k, largest=False, sorted=True)
    
    # 近傍不足対策 (Safety Fallback)
    # 全てinf (同クラス近傍なし) の場合は、SVDを安定させるため「自分自身」のコピーで埋める
    # support_featsの 0~B-1 は x_anchor 自身に対応する
    invalid_mask = (knn_vals == float("inf"))
    self_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(knn_indices)
    final_indices = torch.where(invalid_mask, self_indices, knn_indices)
    
    # --- 4. 局所多様体構成 ---
    # Anchor (B, 1, D)
    anchor_unsqueezed = x_anchor.unsqueeze(1)
    
    # Neighbors (B, k-1, D)
    neighbor_feats = support_feats[final_indices]
    
    # Local Set Z_i: (B, k, D)
    Zi = torch.cat([anchor_unsqueezed, neighbor_feats], dim=1)
    
    # --- 5. SVD & Perturbation ---
    # 中心化
    Zi_mean = Zi.mean(dim=1, keepdim=True)
    Zi_centered = Zi - Zi_mean
    jitter = 1e-4  
    Zi_centered += torch.randn_like(Zi_centered) * jitter
    
    # Batch SVD
    # U: (B, k, k), S: (B, k), Vt: (B, k, D)
    # k < D の場合、full_matrices=Falseなら Sのサイズは (B, k)
    try:
        U, S, Vt = torch.linalg.svd(Zi_centered, full_matrices=False)
    except RuntimeError:
        # SVDが収束しない稀なケースへの対策
        return x_anchor

    # 累積寄与率に基づくスケーリング
    S_sum = S.sum(dim=1, keepdim=True) + 1e-8
    cum_score = torch.cumsum(S, dim=1) / S_sum
    
    # 支配的成分(Dominant)かどうかのマスク
    mask_dominant = cum_score <= rho
    
    # 摂動係数 lambda ~ Beta(alpha, alpha)
    lam = torch.distributions.Beta(alpha, alpha).sample((B, 1)).to(device)
    
    # 支配的成分は 1.0倍(保持)、非支配成分(ノイズ方向)は lambda倍(縮小)
    scale = torch.where(mask_dominant, torch.tensor(1.0, device=device), lam)
    
    # 再構成
    # U @ S_scaled @ Vt + Mean
    S_scaled = S * scale
    reconstructed = U @ (S_scaled.unsqueeze(2) * Vt)
    Zi_new = reconstructed + Zi_mean
    
    # アンカー(index 0)に対応する摂動済み特徴量を抽出
    x_aug = Zi_new[:, 0, :]
    
    return x_aug