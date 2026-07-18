"""Delta-space Generalized Task Arithmetic for LoRA merging.

Pure torch. No intra-package imports (so it is loadable by file path in the
repo's broken test env). Faithfully mirrors mergekit's sparsify + GTA math but
operates directly on full LoRA deltas, avoiding the per-factor squaring and the
meaningless factor-space sign vote of the old path.
"""
from typing import List, Optional

import torch

# ---------------------------------------------------------------- rescale
_RESCALE = ("l1", "l2", "linf")


def _rescaled_masked(tensor: torch.Tensor, mask: torch.Tensor,
                     norm: Optional[str], eps: float = 1e-7) -> torch.Tensor:
    masked = tensor * mask
    if not norm or norm == "none":
        return masked
    if norm == "l1":
        before, after = tensor.abs().sum(), masked.abs().sum()
    elif norm == "l2":
        before, after = tensor.norm(), masked.norm()
    elif norm == "linf":
        before, after = tensor.abs().max(), masked.abs().max()
    else:
        raise ValueError(f"unknown rescale_norm {norm!r}")
    if after < eps:
        return masked
    return masked * (before / after)


# ---------------------------------------------------------------- sparsify
def _magnitude(t, density, rescale_norm):
    if density >= 1:
        return t
    k = int(density * t.numel())
    mask = torch.zeros_like(t)
    w = t.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.argsort(w, descending=True)[:k]
    mask.view(-1)[topk] = 1
    return _rescaled_masked(t, mask, rescale_norm)


def _magnitude_outliers(t, density, rescale_norm, gamma):
    if density >= 1:
        return t
    n = t.numel()
    target_n = int(density * n)
    n_top = int(gamma * n)
    n_bot = n - target_n - n_top
    if n_bot < 0:
        n_top += n_bot
        n_bot = 0
    w = t.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    idx = torch.sort(w, descending=False).indices
    mask = torch.zeros_like(t)
    mask.view(-1)[idx[n_bot:-n_top]] = 1
    return _rescaled_masked(t, mask, rescale_norm)


def _bernoulli(t, density, rescale_norm):
    if density >= 1:
        return t
    work = t.dtype
    if t.device.type == "cpu" and t.dtype in (torch.float16,):
        work = torch.float32
    mask = torch.bernoulli(torch.full_like(t, density, dtype=work)).to(t.dtype)
    return _rescaled_masked(t, mask, rescale_norm)


def _della_magprune(t, density, epsilon, rescale_norm):
    if density >= 1:
        return t
    if density <= 0:
        return torch.zeros_like(t)
    if density + epsilon >= 1 or density - epsilon <= 0:
        raise ValueError("epsilon must keep density +/- epsilon in (0, 1)")
    orig_shape = t.shape
    x = t
    if x.dim() < 2:
        x = x.unsqueeze(0)
    mags = x.abs()
    sorted_idx = torch.argsort(mags, dim=1, descending=False)
    ranks = sorted_idx.argsort(dim=1).to(torch.float32) + 1
    min_r = ranks.min(dim=1, keepdim=True).values
    max_r = ranks.max(dim=1, keepdim=True).values
    rank_norm = ((ranks - min_r) / (max_r - min_r)).clamp(0, 1)
    probs = (density - epsilon) + rank_norm * 2 * epsilon
    mask = torch.bernoulli(probs).to(torch.float32)
    res = _rescaled_masked(x.to(torch.float32), mask, rescale_norm)
    return res.to(t.dtype).reshape(orig_shape)


def sparsify(tensor: torch.Tensor, method: Optional[str], *, density: float = 1.0,
             gamma: float = 0.0, epsilon: float = 0.0,
             rescale_norm: Optional[str] = None) -> torch.Tensor:
    """Sparsify one delta. `method` in {None, magnitude, random,
    magnitude_outliers, della_magprune}. None returns the tensor unchanged."""
    if method is None:
        return tensor
    if method == "magnitude":
        return _magnitude(tensor, density, rescale_norm)
    if method == "magnitude_outliers":
        return _magnitude_outliers(tensor, density, rescale_norm, gamma)
    if method == "random":
        return _bernoulli(tensor, density, rescale_norm)
    if method == "della_magprune":
        return _della_magprune(tensor, density, epsilon, rescale_norm)
    raise ValueError(f"unknown sparsify method {method!r}")


# --------------------------------------------------------- sign + merge
def elect_sign(weighted_deltas: torch.Tensor) -> torch.Tensor:
    """Per-element elected sign from stacked weighted deltas (shape [N, *]).
    Uses the TIES 'sum' method: sign of the summed weighted delta."""
    sign_weight = weighted_deltas.sum(dim=0)
    return (sign_weight >= 0).to(weighted_deltas.dtype) * 2 - 1


def disjoint_merge(deltas: torch.Tensor, weights: torch.Tensor, *,
                   sign_consensus: bool, normalize: bool) -> torch.Tensor:
    """Merge stacked deltas (shape [N, *]) with per-LoRA `weights` (shape [N]).

    weighted_deltas = deltas * weights (broadcast). When sign_consensus, elect a
    per-element sign and keep only agreeing contributions. `normalize` divides by
    the per-element sum of surviving weights (weighted average)."""
    w = weights.clone()
    while w.dim() < deltas.dim():
        w = w.unsqueeze(-1)
    weighted = deltas * w

    if sign_consensus:
        sign = elect_sign(weighted)
        agree = (torch.sign(weighted) == sign).to(deltas.dtype)
    else:
        agree = torch.ones_like(weighted)

    mixed = (weighted * agree).sum(dim=0)
    divisor = (w.abs() * agree).sum(dim=0) if sign_consensus else (w * agree).sum(dim=0)
    divisor = torch.where(divisor.abs() < 1e-8, torch.ones_like(divisor), divisor)
    if normalize:
        mixed = mixed / divisor
    return mixed