# --- SK-Mixup (KernelMixup) dependencies ---
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch import Tensor

def beta_warping(x, alpha: float = 1.0, eps: float = 1e-12, inverse_cdf: bool = False) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.array(x)
    """ベータ分布のCDF/逆CDF でワーピング"""
    if inverse_cdf:
        return scipy.stats.beta.ppf(x, a=alpha + eps, b=alpha + eps)
    else:
        return scipy.stats.beta.cdf(x, a=alpha + eps, b=alpha + eps)

def sim_gauss_kernel(dist, tau_max: float = 1.0, tau_std: float = 0.5, inverse_cdf=False) -> np.ndarray:
    """距離→ガウスカーネル→（逆）CDF パラメータ"""
    dist_rate = tau_max * np.exp(-(dist - 1) / (2 * tau_std * tau_std))
    if inverse_cdf:
        return dist_rate
    return 1.0 / (dist_rate + 1e-12)

class AbstractMixup:
    def __init__(self, alpha: float = 1.0, mode: str = "batch", num_classes: int = 1000) -> None:
        self.alpha = alpha
        self.num_classes = num_classes
        self.mode = mode

    def _get_params(self, batch_size: int, device: torch.device):
        if self.mode == "batch":
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, batch_size)
            ).to(device)
        index = torch.randperm(batch_size, device=device)
        return lam, index

    def _linear_mixing(self, lam, inp: Tensor, index: Tensor) -> Tensor:
        if isinstance(lam, Tensor):
            lam = lam.view(-1, *[1]*(inp.ndim-1)).float()
        return lam * inp + (1 - lam) * inp[index]

    def _mix_target(self, lam, target: Tensor, index: Tensor) -> Tensor:
        y1 = F.one_hot(target, self.num_classes).float()
        y2 = F.one_hot(target[index], self.num_classes).float()
        if isinstance(lam, Tensor):
            lam = lam.view(-1, *[1]*(y1.ndim-1)).float()
        return lam * y1 + (1 - lam) * y2

class KernelMixup(AbstractMixup):
    """Similarity Kernel Mixup (SK-Mixup)"""
    def __init__(
        self,
        alpha: float = 1.0,
        mode: str = "batch",
        num_classes: int = 1000,
        warping: str = "beta_cdf",
        tau_max: float = 1.0,
        tau_std: float = 0.5,
        lookup_size: int = 4096,
    ) -> None:
        super().__init__(alpha, mode, num_classes)
        self.warping = warping
        self.tau_max = tau_max
        self.tau_std = tau_std
        self.lookup_size = lookup_size
        self.rng_gen = None

    def _init_lookup_table(self, device):
        self.rng_gen = torch.distributions.Beta(
            torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
        )
        # 略：lookup 用テーブル構築（必要なら）

    def _get_params(self, batch_size: int, device: torch.device):
        if self.warping == "lookup":
            if self.rng_gen is None:
                self._init_lookup_table(device)
            lam = self.rng_gen.sample_n(batch_size)
        else:
            lam = np.random.beta(self.alpha, self.alpha) if self.mode=="batch" \
                  else np.random.beta(self.alpha, self.alpha, batch_size)
        index = torch.randperm(batch_size, device=device)
        return lam, index

    def __call__(self, x: Tensor, y: Tensor, feats: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size(0), x.device)
        # 距離計算 & warp_param
        if self.warping!="lookup":
            dist = (feats - feats[index]).pow(2).sum(dim=1).detach().cpu().numpy()
            dist /= dist.mean()
            if self.warping in ("inverse_beta_cdf", "no_warp"):
                warp_p = sim_gauss_kernel(dist, self.tau_max, self.tau_std, inverse_cdf=True)
            else:
                warp_p = sim_gauss_kernel(dist, self.tau_max, self.tau_std)
        # λ のワーピング
        if self.warping=="inverse_beta_cdf":
            warped = beta_warping(lam, warp_p, inverse_cdf=True)
            # k_lam = beta_warping(lam, warp_p, inverse_cdf=True)
        else:
            warped = beta_warping(lam, warp_p, inverse_cdf=False)
            # k_lam = beta_warping(lam, warp_p, inverse_cdf=False)
        # k_lam = torch.tensor(k_lam, device=x.device).float().view(-1, *[1]*(x.ndim-1))
        k_lam = torch.tensor(warped, device=x.device, dtype=torch.float32)
        k_lam = k_lam.view(-1, *[1]*(x.ndim-1))
        # ミックス
        mixed_x = self._linear_mixing(k_lam, x, index)
        mixed_y = self._mix_target(k_lam, y, index)
        return mixed_x, mixed_y
