import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm

# ==========================================
# 1. Helper Functions for Augmentations
# ==========================================

def rand_bbox(size, lam):
    """標準的なCutMix用のbbox生成関数"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# --- FMix Implementation ---
def fmix_data(data, alpha=1, decay_power=3, shape=(32, 32), max_soft=0.0, reformulate=False):
    """
    FMix: Fourier Mix
    論文: FMix: Enhancing Mixed Sample Data Augmentation
    """
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    indices = torch.randperm(data.size(0)).to(data.device)
    shuffled_data = data[indices]
    
    mask = torch.from_numpy(mask).float().to(data.device)
    
    # 画像サイズに合わせてマスクを調整
    if mask.shape != data.shape[2:]:
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=data.shape[2:], mode='nearest').squeeze()
        
    x1 = mask * data
    x2 = (1 - mask) * shuffled_data
    return x1 + x2, indices, lam

def sample_mask(alpha, decay_power, shape, max_soft, reformulate):
    """FMix用の周波数マスク生成"""
    if isinstance(shape, int):
        shape = (shape, shape)

    # Low frequency noise generation
    d = shape[0]
    c = shape[1]
    
    lam = np.random.beta(alpha, alpha)
    
    mask = np.zeros(shape)
    
    # Frequency domain filtering
    x = np.arange(-d // 2, d // 2)
    y = np.arange(-c // 2, c // 2)
    xx, yy = np.meshgrid(x, y)
    
    # Distance from center
    dist = np.sqrt(xx**2 + yy**2) + 1e-10 # avoid div by zero
    
    # Filter with decay power
    filter_mask = 1.0 / (dist ** decay_power)
    
    # Complex noise
    noise = np.random.randn(d, c) + 1j * np.random.randn(d, c)
    noise = np.fft.fftshift(noise) # Shift zero freq to center
    noise = noise * filter_mask
    noise = np.fft.ifftshift(noise) # Shift back
    
    img = np.fft.ifft2(noise).real
    
    # Normalize
    img = (img - img.min()) / (img.max() - img.min() + 1e-7)
    
    # --- 【修正箇所】 ---
    # Binarize based on lambda
    flat = np.sort(img.flatten())
    
    # lam の割合だけ「1」を残したい場合、
    # 閾値は「下位 (1-lam)」の位置にする必要があります。
    # 例: lam=0.9 なら、下位10% (0.1) を境界線にして、それ以上を1にする
    idx = int((1 - lam) * len(flat))
    
    # インデックスのエラー防止
    idx = np.clip(idx, 0, len(flat) - 1)
    
    threshold = flat[idx]
    
    # これで lam=0.9 のとき、約90%の画素が True(1) になります
    binary_mask = (img > threshold).astype(float)
    
    return lam, binary_mask

# --- ResizeMix Implementation ---
def resizemix_data(data, labels, alpha=1.0, device='cuda'):
    """
    ResizeMix: Mixing Data with Preserved Object Information
    """
    indices = torch.randperm(data.size(0)).to(device)
    shuffled_data = data[indices]
    y_a, y_b = labels, labels[indices]

    # lambda sampling (ResizeMixでは一様分布やBeta分布などが議論されるが、一般的にCutMix同様Betaを使用)
    # 論文ではタスク依存だが、ここでは標準的な実装を採用
    lam = np.random.beta(alpha, alpha)
    
    # リサイズスケールの決定 (論文推奨: 0.1 ~ 0.8程度)
    # ここではlam面積比になるようにスケールを逆算する実装と、単純にリサイズする実装があるが、
    # 論文の "random paste" に従い、リサイズ画像をランダム位置に貼る。
    
    batch_size, C, H, W = data.shape
    
    # ResizeMixの定義: 画像Bをリサイズして画像Aに貼る
    # 貼られる画像(B)のスケールτ
    scale = np.sqrt(1.0 - lam) # 1-lam がBの面積比率になるように設定
    scale = np.clip(scale, 0.1, 0.9) # 極端なサイズを回避
    
    new_H, new_W = int(H * scale), int(W * scale)
    
    # 画像Bをリサイズ
    resize_B = F.interpolate(shuffled_data, size=(new_H, new_W), mode='bilinear', align_corners=False)
    
    # 貼り付け位置
    rx = np.random.randint(0, H - new_H + 1)
    ry = np.random.randint(0, W - new_W + 1)
    
    mixed_data = data.clone()
    mixed_data[:, :, rx:rx+new_H, ry:ry+new_W] = resize_B
    
    # ラムダの再計算 (正確なピクセル比率)
    real_lam = 1.0 - (new_H * new_W) / (H * W)
    
    return mixed_data, y_a, y_b, real_lam

# --- SaliencyMix Implementation ---
def saliencymix_data(model, data, labels, criterion, alpha=1.0, device='cuda'):
    """
    SaliencyMix: 勾配を利用して重要な領域を特定しCutMixを行う
    """
    model.eval() # 勾配取得のため一度evalモード推奨だが、BatchNormの挙動維持のためtrainのままにする手もある。ここでは安全に勾配だけ取る。
    
    # 入力に対する勾配を有効化
    data_copy = data.clone().detach()
    data_copy.requires_grad = True
    
    # 予測とバックプロパゲーション（勾配取得）
    preds = model(data_copy, labels=None, device=device, augment="None", aug_ok=False) # augment="None"で純粋な推論を行う想定
    loss = criterion(preds, labels)
    loss.backward()
    
    # Saliency Map: チャンネル方向の最大絶対値勾配
    saliency = torch.max(data_copy.grad.data.abs(), dim=1)[0] # [Batch, H, W]
    
    # インデックスのシャッフル
    indices = torch.randperm(data.size(0)).to(device)
    shuffled_data = data[indices]
    y_a, y_b = labels, labels[indices]
    
    lam = np.random.beta(alpha, alpha)
    
    # Saliencyに基づいてbboxを決定
    # 各画像の最もSaliencyが高い座標を取得
    B, H, W = saliency.shape
    saliency_flat = saliency.view(B, -1)
    idx_max = torch.argmax(saliency_flat, dim=1) # [Batch]
    cx = idx_max // W
    cy = idx_max % W
    
    # CutMix用の矩形サイズ
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    mixed_data = data.clone()
    
    # バッチ内の各画像に対して処理（ベクトル化推奨だが、座標が個別に違うためループ処理が見通し良い）
    # ※高速化のためにはベクトル化が必要だが、ここでは可読性と正確性重視
    for i in range(B):
        _cx = cx[i].item()
        _cy = cy[i].item()
        
        bbx1 = np.clip(_cx - cut_w // 2, 0, W).astype(int)
        bby1 = np.clip(_cy - cut_h // 2, 0, H).astype(int)
        bbx2 = np.clip(_cx + cut_w // 2, 0, W).astype(int)
        bby2 = np.clip(_cy + cut_h // 2, 0, H).astype(int)
        
        mixed_data[i, :, bbx1:bbx2, bby1:bby2] = shuffled_data[i, :, bbx1:bbx2, bby1:bby2]

    # 正確なlamを計算（bboxが端にかかって小さくなる場合があるため）
    # 簡易的に固定lamを返す実装が多いが、厳密には画素数比
    real_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    
    return mixed_data, y_a, y_b, real_lam

# --- PuzzleMix Implementation (Simplified Greedy Strategy) ---
def puzzlemix_data(model, data, labels, criterion, alpha=1.0, device='cuda', block_size=4):
    """
    PuzzleMix: Saliency情報を利用し、最適な輸送計画に近い形でパッチを混ぜる
    ※ 完全な最適化は遅すぎるため、Greedyな簡易版(Block-based)を実装
    """
    model.eval()
    data_copy = data.clone().detach()
    data_copy.requires_grad = True
    
    preds = model(data_copy, labels=None, device=device, augment="None", aug_ok=False)
    loss = criterion(preds, labels)
    loss.backward()
    
    # [Batch, H, W]
    saliency = torch.max(data_copy.grad.data.abs(), dim=1)[0] 
    
    indices = torch.randperm(data.size(0)).to(device)
    shuffled_data = data[indices]
    y_a, y_b = labels, labels[indices]
    lam = np.random.beta(alpha, alpha)
    
    mixed_data = data.clone()
    
    # 画像をブロックに分割して処理 (例: 32x32 -> 4x4のブロックが8x8個)
    B, C, H, W = data.shape
    n_rows = H // block_size
    n_cols = W // block_size
    
    # Saliencyをブロック単位に平均プーリング
    s_a = F.avg_pool2d(saliency, block_size) # [B, n_rows, n_cols]
    s_b = F.avg_pool2d(saliency[indices], block_size)
    
    # 各画像で「残すべきブロック数」を決定 (lamに基づく)
    num_blocks = n_rows * n_cols
    keep_n = int(lam * num_blocks)
    
    # Greedy Strategy:
    # 画像Aの重要な部分と、画像Bの重要な部分が重ならないように混ぜるのが理想だが
    # 簡易実装として「画像Bの最も重要な部分」を「画像Aの最も重要でない部分」に置換、あるいは
    # 画像Aの上位lam%を残し、残りを画像Bの対応箇所で埋める（これはSaliencyMixに近い）。
    # PuzzleMixの核心は「場所の移動（Transport）」だが、実装コスト上、
    # ここでは「画像Aと画像BのSaliency情報の和が最大化されるようなマスク生成」を行う。
    
    # Z = mask. 1ならAを採用、0ならBを採用。
    # 「AのSaliency - BのSaliency」が大きい場所ほどAを残したい
    diff = s_a - s_b # [B, n_rows, n_cols]
    
    # 上位 keep_n 個のブロックを1にする
    diff_flat = diff.view(B, -1)
    vals, idx = torch.topk(diff_flat, keep_n, dim=1)
    
    mask = torch.zeros_like(diff_flat)
    mask.scatter_(1, idx, 1.0)
    mask = mask.view(B, n_rows, n_cols)
    
    # マスクを元の解像度にアップサンプリング
    mask_up = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')
    
    mixed_data = data * mask_up + shuffled_data * (1 - mask_up)
    
    # 実際の混合比率
    real_lam = mask_up.mean(dim=(1, 2, 3)) # Batchごとに異なるlam
    
    # mixup_criterionはスカラーlamを想定している場合が多いが、
    # バッチごとにlamが違う場合は対応が必要。ここでは平均またはベクトルを返す。
    # ユーザーのmixup_criterion実装に依存するが、一般化のため平均を返すか、呼び出し側で調整。
    # ここでは、呼び出し側で処理しやすいよう、ベクトルを返す設計にするが、
    # 既存コードとの互換性のため平均値を代表値として返す（精度重視ならLoss側での変更が必要）。
    avg_lam = real_lam.mean().item()
    
    return mixed_data, y_a, y_b, avg_lam