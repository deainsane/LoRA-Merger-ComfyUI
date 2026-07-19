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


# Rows above this are processed in blocks (below, in one shot). The whole-tensor
# path is kept for small tensors so it stays bit-for-bit identical to mergekit's
# della (same bernoulli RNG order) -- the parity unit test relies on that.
_DELLA_CHUNK_ROWS = 4096


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
    if x.shape[0] <= _DELLA_CHUNK_ROWS:
        res = _della_whole(x, density, epsilon, rescale_norm)
    else:
        res = _della_chunked(x, density, epsilon, rescale_norm)
    return res.reshape(orig_shape)


def _della_whole(x, density, epsilon, rescale_norm):
    """Original single-shot della: bit-for-bit matches mergekit for a given RNG
    seed. Materializes several full-size temporaries (two argsorts etc.), so it
    is only used for tensors small enough not to matter."""
    mags = x.abs()
    sorted_idx = torch.argsort(mags, dim=1, descending=False)
    ranks = sorted_idx.argsort(dim=1).to(torch.float32) + 1
    min_r = ranks.min(dim=1, keepdim=True).values
    max_r = ranks.max(dim=1, keepdim=True).values
    rank_norm = ((ranks - min_r) / (max_r - min_r)).clamp(0, 1)
    probs = (density - epsilon) + rank_norm * 2 * epsilon
    mask = torch.bernoulli(probs).to(torch.float32)
    res = _rescaled_masked(x.to(torch.float32), mask, rescale_norm)
    return res.to(x.dtype)


def _norm_value(t: torch.Tensor, norm: Optional[str]):
    """Scalar norm used for global rescale, or None if no rescale applies."""
    if not norm or norm == "none":
        return None
    if norm == "l1":
        return t.abs().sum()
    if norm == "l2":
        return t.norm()
    if norm == "linf":
        return t.abs().max()
    raise ValueError(f"unknown rescale_norm {norm!r}")


def _della_chunked(x, density, epsilon, rescale_norm, chunk_rows=_DELLA_CHUNK_ROWS):
    """Memory-frugal della for large layers (e.g. FLUX [21504, 3072]).

    della's ranking is per-row (``argsort(dim=1)``), so rows are independent and
    processing them in blocks is exact -- only the bernoulli draw order changes,
    and della is stochastic pruning anyway. The whole-tensor path peaks at ~11x
    the delta (two full-layer int64 argsorts + a stack of fp32 temporaries); this
    keeps every temporary to a single ``chunk_rows`` block, so peak is a small
    constant regardless of layer size. The mask is applied in place; the global
    rescale (l1/l2/linf) is computed across the whole tensor to match semantics."""
    rows, cols = x.shape
    in_place = x.dtype == torch.float32
    out = x if in_place else x.to(torch.float32)  # in-place on fp32 input; else one copy
    before = _norm_value(out, rescale_norm)        # from original values, pre-mask
    denom = float(cols - 1) if cols > 1 else 1.0
    two_eps = 2.0 * epsilon
    base = density - epsilon
    for lo in range(0, rows, chunk_rows):
        hi = min(lo + chunk_rows, rows)
        blk = out[lo:hi]                            # view into out
        sorted_idx = torch.argsort(blk.abs(), dim=1, descending=False)
        ranks = sorted_idx.argsort(dim=1)          # 0..cols-1 per row
        del sorted_idx
        probs = base + (ranks.to(torch.float32) / denom) * two_eps
        del ranks
        blk.mul_(torch.bernoulli(probs))           # mask in place
        del probs
    if before is not None:
        after = _norm_value(out, rescale_norm)
        if after >= 1e-7:
            out.mul_(before / after)
    return out.to(x.dtype)


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


# --------------------------------------------------------- mode config
_MODE_SPARSIFY = {
    "linear": None,
    "task_arithmetic": None,
    "ties": "magnitude",
    "dare": "random",
    "della": "della_magprune",
    "breadcrumbs": "magnitude_outliers",
}
_MODE_ALWAYS_CONSENSUS = {"ties"}
_MODE_NEVER_CONSENSUS = {"linear", "task_arithmetic"}
_MODE_DEFAULT_RESCALE = {
    "linear": False,
    "task_arithmetic": False,
    "ties": False,
    "dare": False,
    "della": True,
    "breadcrumbs": False,
}

GTA_MODES = tuple(_MODE_SPARSIFY.keys())


def resolve_rescale_norm(mode: str, rescale_norm: str) -> Optional[str]:
    if rescale_norm == "default":
        return "l1" if _MODE_DEFAULT_RESCALE.get(mode, False) else None
    if rescale_norm == "none":
        return None
    if rescale_norm in _RESCALE:
        return rescale_norm
    raise ValueError(f"unknown rescale_norm {rescale_norm!r}")


def gta_merge(deltas: List[torch.Tensor], weights: torch.Tensor, *, mode: str,
              normalize: bool = True, density: float = 1.0, epsilon: float = 0.0,
              gamma: float = 0.0, sign_consensus_algorithm: bool = False,
              rescale_norm: str = "default") -> torch.Tensor:
    """Merge a list of full LoRA deltas with GTA semantics on the delta itself.

    `weights` are the per-LoRA merge strengths (signed). Returns the merged delta.

    NOTE: this **consumes** `deltas` -- entries are replaced/freed in place to keep
    peak memory low (large FLUX deltas are hundreds of MB each). Snapshot anything
    you need to reuse before calling."""
    if mode not in _MODE_SPARSIFY:
        raise ValueError(f"unknown GTA mode {mode!r}")
    if mode in _MODE_ALWAYS_CONSENSUS:
        consensus = True
    elif mode in _MODE_NEVER_CONSENSUS:
        consensus = False
    else:
        consensus = bool(sign_consensus_algorithm)

    sp = _MODE_SPARSIFY[mode]
    res = resolve_rescale_norm(mode, rescale_norm)
    # Sparsify in place (replace each entry) so we never hold both the original
    # and sparsified copy of every delta at once.
    for i in range(len(deltas)):
        deltas[i] = sparsify(deltas[i], sp, density=density, gamma=gamma,
                             epsilon=epsilon, rescale_norm=res)
    w = weights.to(deltas[0].dtype)
    return _stream_merge(deltas, w, sign_consensus=consensus, normalize=normalize)


def _stream_merge(deltas: List[torch.Tensor], weights: torch.Tensor, *,
                  sign_consensus: bool, normalize: bool) -> torch.Tensor:
    """Memory-frugal merge over a *list* of deltas.

    Never stacks into an ``[N, out, in]`` tensor (that plus mergekit-style
    intermediate copies peaked at ~13x a single delta for large FLUX layers and
    exhausted VRAM). Instead it streams: one accumulator for the elected sign,
    then per-element accumulators for the numerator and divisor. Peak is a small
    constant number of ``[out, in]`` buffers, independent of ``N``."""
    n = len(deltas)

    if sign_consensus:
        acc = deltas[0] * weights[0]
        for i in range(1, n):
            acc = acc + deltas[i] * weights[i]
        # Elected sign as +1/-1 (matches mergekit's TIES 'sum' method). Using a
        # 3-state sign comparison below means sparsified-out elements (exact 0,
        # sign 0) never match the elected +/-1 and so are excluded from both the
        # numerator and the divisor -- essential when density < 1.
        sign_pm = (acc >= 0).to(deltas[0].dtype) * 2 - 1
        del acc

    mixed = None
    divisor = None
    zero = None
    for i in range(n):
        wd = deltas[i] * weights[i]
        deltas[i] = None                    # free the input delta as soon as used
        if sign_consensus:
            if zero is None:
                zero = torch.zeros((), dtype=wd.dtype, device=wd.device)
            agree = torch.sign(wd) == sign_pm
            if normalize:
                dcontrib = agree.to(wd.dtype) * weights[i].abs()
                divisor = dcontrib if divisor is None else divisor + dcontrib
                del dcontrib
            wd = torch.where(agree, wd, zero)   # reuse wd as the masked contribution
            del agree
        elif normalize:
            divisor = weights[i] if divisor is None else divisor + weights[i]
        mixed = wd if mixed is None else mixed + wd
        del wd

    if normalize:
        if not sign_consensus:
            divisor = torch.as_tensor(divisor, dtype=mixed.dtype, device=mixed.device)
            divisor = divisor.expand_as(mixed) if divisor.dim() == 0 else divisor
        divisor = torch.where(divisor.abs() < 1e-8, torch.ones_like(divisor), divisor)
        mixed = mixed / divisor
    return mixed


# --------------------------------------------------------- sign + merge
def elect_sign(weighted_deltas: torch.Tensor) -> torch.Tensor:
    """Per-element elected sign from stacked weighted deltas (shape [N, *]).
    Uses the TIES 'sum' method: sign of the summed weighted delta."""
    sign_weight = weighted_deltas.sum(dim=0)
    return (sign_weight >= 0).to(weighted_deltas.dtype) * 2 - 1


def disjoint_merge(deltas, weights: torch.Tensor, *,
                   sign_consensus: bool, normalize: bool) -> torch.Tensor:
    """Merge deltas with per-LoRA ``weights`` (shape [N]).

    Accepts either a list of ``[out, in]`` deltas or a stacked ``[N, *]`` tensor
    (the latter is unbound into views, no copy). Delegates to the memory-frugal
    :func:`_stream_merge`; kept as the stable public entry used by the unit
    tests. When ``sign_consensus``, elect a per-element sign and keep only
    agreeing contributions; ``normalize`` divides by the per-element sum of
    surviving weights."""
    if isinstance(deltas, torch.Tensor):
        deltas = list(deltas.unbind(0))
    return _stream_merge(deltas, weights, sign_consensus=sign_consensus,
                         normalize=normalize)