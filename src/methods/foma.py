import torch
import torch.nn.functional as F

def foma(X, Y, num_classes, alpha, rho, small_singular=True, lam=None):
    """
    FOMA for image classification tasks.
    X: Input images, shape [B, C, H, W]
    Y: Labels, shape [B] or [B, num_classes]
    """
    B = X.shape[0]
    # Flatten image to [B, C*H*W]
    X_flat = X.view(B, -1)

    # Convert labels to one-hot if needed
    if Y.ndim == 1:  # [B]
        Y_onehot = F.one_hot(Y, num_classes=num_classes).float()
    else:
        Y_onehot = Y.float()

    # Concatenate X and Y
    Z = torch.cat([X_flat, Y_onehot], dim=1)

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

    # Optionally: Convert one-hot back to class labels (argmax)
    Y_onehot_scaled = torch.clamp(Y_onehot_scaled, min=0)
    
    # 各サンプルごとに合計を計算
    sum_per_sample = Y_onehot_scaled.sum(dim=1, keepdim=True)
    
    normalized_labels = torch.where(
    sum_per_sample == 0,
    torch.ones_like(Y_onehot_scaled) / Y_onehot_scaled.size(1),  # 全要素0なら均等分布
    Y_onehot_scaled / sum_per_sample
)

    return X_scaled, normalized_labels


def compute_foma_loss(model, images, labels, lambda_almp=1.0, device='cuda'):
    model.train()
    images = images.to(device)
    labels = labels.to(device)

    # 特徴抽出
    features = model.extract_features(images)  # (B, D)

    # 元の分類出力
    logits_orig = model.linear(features)
    loss_orig = F.cross_entropy(logits_orig, labels)

    # FOMAによる特徴摂動
    # features_almp = local_pca_perturbation_torch(features, method=method)
    features_foma, labels_foma = foma(features, labels, num_classes=100, alpha=1.0, rho=0.9)
    logits_almp = model.linear(features_foma)
    loss_almp = F.cross_entropy(logits_almp, labels_foma)

    return loss_orig + lambda_almp * loss_almp, logits_orig