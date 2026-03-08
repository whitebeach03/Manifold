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

    B, D = x_anchor.shape
    device = x_anchor.device
    
    if y_anchor.ndim > 1:
        y_idx = y_anchor.argmax(dim=1)
    else:
        y_idx = y_anchor
        
    mem_feats, mem_labels = memory_bank.get_memory()
    
    if mem_feats is None:
        return x_anchor

    support_feats = torch.cat([x_anchor.detach(), mem_feats], dim=0) # (B+M, D)
    support_labels = torch.cat([y_idx, mem_labels], dim=0)           # (B+M,)

    dist = torch.cdist(x_anchor, support_feats)

    class_mask = (y_idx.unsqueeze(1) == support_labels.unsqueeze(0))
    
    dist.masked_fill_(~class_mask, float("inf"))

    search_k = k - 1

    search_k = min(search_k, support_feats.shape[0])
    
    if search_k < 1:
        return x_anchor

    knn_vals, knn_indices = torch.topk(dist, k=search_k, largest=False, sorted=True)
    
    invalid_mask = (knn_vals == float("inf"))
    self_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(knn_indices)
    final_indices = torch.where(invalid_mask, self_indices, knn_indices)

    anchor_unsqueezed = x_anchor.unsqueeze(1)

    neighbor_feats = support_feats[final_indices]

    Zi = torch.cat([anchor_unsqueezed, neighbor_feats], dim=1)

    Zi_mean = Zi.mean(dim=1, keepdim=True)
    Zi_centered = Zi - Zi_mean
    jitter = 1e-4  
    Zi_centered += torch.randn_like(Zi_centered) * jitter

    try:
        U, S, Vt = torch.linalg.svd(Zi_centered, full_matrices=False)
    except RuntimeError:
        return x_anchor

    S_sum = S.sum(dim=1, keepdim=True) + 1e-8
    cum_score = torch.cumsum(S, dim=1) / S_sum

    mask_dominant = cum_score <= rho

    lam = torch.distributions.Beta(alpha, alpha).sample((B, 1)).to(device)

    scale = torch.where(mask_dominant, torch.tensor(1.0, device=device), lam)

    S_scaled = S * scale
    reconstructed = U @ (S_scaled.unsqueeze(2) * Vt)
    Zi_new = reconstructed + Zi_mean

    x_aug = Zi_new[:, 0, :]
    
    return x_aug