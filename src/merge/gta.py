"""Delta-space Generalized Task Arithmetic for LoRA merging.

Pure torch. No intra-package imports (so it is loadable by file path in the
repo's broken test env). Faithfully mirrors mergekit's sparsify + GTA math but
operates directly on full LoRA deltas, avoiding the per-factor squaring and the
meaningless factor-space sign vote of the old path.
"""
import math
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
# Above this many elements, magnitude / magnitude_outliers selection picks its
# threshold from a GPU histogram of |t| instead of a full argsort. A full argsort
# materializes an int64 permutation (8 bytes/elem == 2x an fp32 delta) plus sort
# workspace -- on a giant FLUX layer that single allocation OOMs an 8 GB card on
# top of the resident deltas (this is what broke TIES and Breadcrumbs). A host
# k-th-value avoids the OOM but is ~300x slower (a full PCIe copy + CPU select,
# per delta per key, serialized -- what made TIES/Breadcrumbs unusably slow). The
# histogram is a single O(numel) GPU pass, allocates only a few-KB histogram, and
# lands within ~1e-3 of the target density. Small tensors keep the exact GPU
# top-k so they stay bit-for-bit identical to mergekit (the parity tests rely on
# that).
_SPARSIFY_EXACT_ELEMS = 16 * 1024 * 1024
_SPARSIFY_HIST_BINS = 8192


def _topk_hist_threshold(w_abs: torch.Tensor, keep_k: int) -> float:
    """Magnitude threshold ``tau`` such that ``w_abs >= tau`` keeps ~``keep_k`` of
    the largest elements, found from a GPU histogram of ``w_abs`` (no sort, no host
    transfer). ``w_abs`` is non-negative. Returns a python float."""
    numel = w_abs.numel()
    if keep_k <= 0:
        return float(w_abs.max()) + 1.0     # keep nothing
    if keep_k >= numel:
        return 0.0                          # keep everything
    hi = float(w_abs.max())
    if hi <= 0.0:
        return 0.0
    hist = torch.histc(w_abs, bins=_SPARSIFY_HIST_BINS, min=0.0, max=hi)
    # Cumulative count from the largest-magnitude bin downward; take the lower edge
    # of the fewest top bins whose running count first reaches keep_k.
    csum = torch.cumsum(hist.flip(0), 0)
    j = int(torch.searchsorted(csum, torch.tensor(float(keep_k), device=w_abs.device)))
    j = min(j, _SPARSIFY_HIST_BINS - 1)
    return hi * (1.0 - (j + 1) / _SPARSIFY_HIST_BINS)


def _magnitude(t, density, rescale_norm):
    if density >= 1:
        return t
    k = int(density * t.numel())
    numel = t.numel()
    if numel <= _SPARSIFY_EXACT_ELEMS:
        mask = torch.zeros_like(t)
        w = t.abs().view(-1)
        if w.device.type == "cpu":
            w = w.float()
        topk = torch.argsort(w, descending=True)[:k]
        mask.view(-1)[topk] = 1
    elif k <= 0:
        mask = torch.zeros_like(t)
    else:
        # Keep the top-k by magnitude via a histogram threshold (~k up to bin
        # resolution, negligible for merge quality).
        w = t.abs()
        tau = _topk_hist_threshold(w, k)
        mask = (w >= tau).to(t.dtype)
        del w
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
    if n <= _SPARSIFY_EXACT_ELEMS:
        w = t.abs().view(-1)
        if w.device.type == "cpu":
            w = w.float()
        idx = torch.sort(w, descending=False).indices
        mask = torch.zeros_like(t)
        mask.view(-1)[idx[n_bot:-n_top]] = 1
    else:
        # Middle band: drop the smallest n_bot and largest n_top by magnitude, via
        # GPU-histogram thresholds (no sort workspace, no host round-trip).
        # tau_low keeps the top (n - n_bot); tau_high keeps the top n_top (dropped).
        w = t.abs()
        mask = torch.ones_like(t)
        if n_bot > 0:
            tau_low = _topk_hist_threshold(w, n - n_bot)
            mask *= (w >= tau_low).to(t.dtype)
        if n_top > 0:
            tau_high = _topk_hist_threshold(w, n_top)
            mask *= (w < tau_high).to(t.dtype)
        del w
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
_DELLA_CHUNK_ROWS = 4096            # whole-vs-chunked threshold (rows)
# Cap the elements per chunk so della's temporaries (an int64 argsort + an int32
# rank buffer) stay a small constant on *wide* layers -- e.g. KREA2 mlp deltas
# are [16384, 6144], where a 4096-row block needs a ~1 GiB contiguous int64
# tensor that fails against a fragmented VRAM pool even with GiB free. Narrow
# layers (small `cols`) are unaffected: they still use the full row block.
_DELLA_CHUNK_ELEMS = 4 * 1024 * 1024


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


def _della_chunked(x, density, epsilon, rescale_norm, chunk_rows=None):
    """Memory-frugal della for large layers (e.g. KREA2 mlp [16384, 6144]).

    della's ranking is per-row (``argsort(dim=1)``), so rows are independent and
    processing them in blocks is exact -- only the bernoulli draw order changes,
    and della is stochastic pruning anyway. The whole-tensor path peaks at ~11x
    the delta (two full-layer int64 argsorts + a stack of fp32 temporaries); this
    keeps every temporary to a single block, so peak is a small constant
    regardless of layer size. The mask is applied in place; the global rescale
    (l1/l2/linf) is computed across the whole tensor to match semantics.

    Two things bound peak VRAM: the block is capped by both a row count and an
    element budget (so *wide* layers get short blocks), and the per-row rank is
    built with a single ``scatter_`` into an int32 buffer instead of a second
    int64 ``argsort`` (which would double the index memory and add sort
    workspace -- the allocation that OOMs on an 8 GB card)."""
    rows, cols = x.shape
    if chunk_rows is None:
        chunk_rows = min(_DELLA_CHUNK_ROWS, max(1, _DELLA_CHUNK_ELEMS // cols))
    in_place = x.dtype == torch.float32
    out = x if in_place else x.to(torch.float32)  # in-place on fp32 input; else one copy
    before = _norm_value(out, rescale_norm)        # from original values, pre-mask
    denom = float(cols - 1) if cols > 1 else 1.0
    two_eps = 2.0 * epsilon
    base = density - epsilon
    # ascending rank positions 0..cols-1, reused for every block via broadcast
    positions = torch.arange(cols, device=out.device, dtype=torch.int32).unsqueeze(0)
    for lo in range(0, rows, chunk_rows):
        hi = min(lo + chunk_rows, rows)
        blk = out[lo:hi]                            # view into out
        sorted_idx = torch.argsort(blk.abs(), dim=1, descending=False)
        # per-row ascending rank of each element: rank[i, sorted_idx[i, j]] = j.
        # Equivalent to sorted_idx.argsort(dim=1) but avoids a second int64
        # tensor + its sort workspace.
        ranks = torch.empty((hi - lo, cols), dtype=torch.int32, device=out.device)
        ranks.scatter_(1, sorted_idx, positions.expand(hi - lo, cols))
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


# Above this many elements, the top-k 'select' threshold is found from a GPU
# histogram rather than an exact top-k. torch.topk on CUDA allocates an O(numel)
# sort workspace (~4 bytes/elem) -- on a giant FLUX layer that single allocation is
# enough to OOM an 8 GB card on top of the resident deltas; a host kthvalue avoids
# the OOM but is ~300x slower (full PCIe copy + CPU select). The histogram (see
# _topk_hist_threshold) is a single O(numel) GPU pass, a few-KB buffer, and lands
# within ~1e-3 of the target density. Small tensors keep the exact GPU top-k.
_SCE_TOPK_GPU_LIMIT = 16 * 1024 * 1024


def _sce_select_mask(var: torch.Tensor, select_topk: float) -> Optional[torch.Tensor]:
    """0/1 mask keeping the ~`select_topk` fraction of highest-variance elements.

    Returns None if the whole tensor is kept, or a zero mask sentinel handling is
    left to the caller (k==0 -> caller returns zeros). `var` is non-negative."""
    numel = var.numel()
    nonzero = int(torch.count_nonzero(var))
    k = int(nonzero * select_topk)
    if k <= 0:
        return var.new_zeros(var.shape)  # nothing selected -> zero delta
    if numel <= _SCE_TOPK_GPU_LIMIT:
        idx = torch.topk(var.view(-1), k=k, largest=True).indices
        flat = var.new_zeros(numel)
        flat[idx] = 1
        return flat.view(var.shape)
    # Large layer: pick the top-k-variance threshold from a GPU histogram (no sort
    # workspace, no host round-trip -- a host kthvalue here is ~300x slower). Ties
    # may keep slightly >k, harmless for SCE's approximate variance selection.
    tau = _topk_hist_threshold(var, k)
    return (var >= tau).to(var.dtype)


def karcher_delta_merge(deltas: List[torch.Tensor], max_iter: int = 10,
                        tol: float = 1e-5) -> torch.Tensor:
    """Memory-frugal Riemannian (Karcher) mean of a *list* of deltas.

    Numerically identical to mergekit's ``karcher_merge_tensors`` with equal
    weights (verified rel < 1e-6), but it never makes the ~3N full-tensor copies
    the stock path does (``apply_weights_to_tensors`` -> N, unit vectors -> N, a
    per-iteration ``ui - dot*u`` transient), which OOMs a large FLUX layer on an
    8 GB card. Instead it **consumes** ``deltas`` (normalizes each entry in place
    into its unit vector) and accumulates the tangent with scalar-alpha in-place
    adds, so peak is the N unit tensors plus two ``[out, in]`` accumulators.
    """
    n = len(deltas)
    if n == 0:
        raise ValueError("karcher_delta_merge requires at least one delta")
    if n == 1:
        return deltas[0]
    dtype, device, shape = deltas[0].dtype, deltas[0].device, deltas[0].shape

    # Norms of the originals (for the final global scale), then normalize in place
    # into unit vectors -- consuming the inputs so we never hold originals + units.
    norms = []
    units = []
    for i in range(n):
        t = deltas[i]
        deltas[i] = None
        nrm = torch.linalg.norm(t.float()).item()
        norms.append(nrm)
        if nrm > 0.0:
            t.div_(nrm)
            units.append(t)
        # zero-norm deltas contribute nothing to the direction (and 0 to the scale)
    if not units:
        return torch.zeros(shape, dtype=dtype, device=device)

    m = len(units)
    a = 1.0 / m                       # equal weights over the valid units

    # Initial guess: normalized arithmetic mean of the unit vectors.
    u = torch.zeros_like(units[0])
    for ui in units:
        u.add_(ui, alpha=a)
    norm_u = torch.linalg.norm(u.float()).item()
    if norm_u < tol:
        u = units[0].clone()
    else:
        u.div_(norm_u)

    # Iterative Karcher mean on the hypersphere.
    for _ in range(max_iter):
        T = torch.zeros_like(u)
        for ui in units:
            dot = float(torch.clamp(torch.dot(u.flatten(), ui.flatten()), -1.0, 1.0))
            theta = math.acos(dot)
            if theta < tol:
                continue
            coeff = a * (theta / math.sin(theta))
            # T += coeff * (ui - dot*u), done as two scalar-alpha in-place adds
            # (no full-size ``ui - dot*u`` temporary).
            T.add_(ui, alpha=coeff)
            T.add_(u, alpha=-coeff * dot)
        norm_T = torch.linalg.norm(T.float()).item()
        if norm_T < tol:
            break
        # u = cos(||T||)*u + sin(||T||)*(T/||T||), in place.
        u.mul_(math.cos(norm_T)).add_(T, alpha=math.sin(norm_T) / norm_T)
        u_norm = torch.linalg.norm(u.float()).item()
        if u_norm > tol:
            u.div_(u_norm)

    # Global scale: equal-weight mean of the ORIGINAL norms (all n tensors).
    s = sum(nrm for nrm in norms) / n
    return u.mul_(s)


def sce_delta_merge(deltas: List[torch.Tensor], select_topk: float,
                    normalize: bool = True) -> torch.Tensor:
    """Memory-frugal SCE (Select-Calculate-Erase) merge over a *list* of deltas.

    Faithful to mergekit's ``sce_merge`` with a zero base tensor, but it never
    stacks the deltas into an ``[N, out, in]`` tensor and never allocates a
    full-layer GPU sort workspace -- either of those, on top of the N resident
    deltas, OOMs an 8 GB card on large FLUX layers. It keeps the same footprint as
    the GTA stream merge: the N input deltas plus a small constant number of
    ``[out, in]`` buffers (freed/consumed in place), independent of N.

    Steps (streamed): one pass for the per-element sum and sum-of-squares (variance
    for the 'select' mask + elected sign), then -- for the merge -- a sign-consensus
    accumulate over the selected task vectors.

    ``normalize`` picks the merge convention (both keep the same select + elected
    sign):
      * ``True``  -> mergekit's SCE: a normalized weighted AVERAGE (per-tensor
        variance weights, divided by the surviving weight sum). A blend; magnitude
        is bounded by a single delta regardless of how many LoRAs stack.
      * ``False`` -> additive SUM of the sign-agreeing selected contributions, so
        per-LoRA strengths act as gains and stacked LoRAs keep full magnitude
        (matches ComfyUI stacking and the rest of the node suite). This is the
        default the SCE node exposes.

    ``deltas`` are the already strength-weighted task vectors and are **consumed**
    (masked/freed in place)."""
    n = len(deltas)
    if n == 0:
        raise ValueError("sce_delta_merge requires at least one delta")
    dtype, device, shape = deltas[0].dtype, deltas[0].device, deltas[0].shape

    # Single pass: sum (s1) and sum-of-squares (s2). Only two full-layer buffers
    # are held here (plus one transient square per iter), vs. the previous
    # s1+mean+var+var.abs() quartet.
    s1 = torch.zeros(shape, dtype=dtype, device=device)
    s2 = torch.zeros(shape, dtype=dtype, device=device)
    for d in deltas:
        s1 += d
        s2 += d * d
    # var = E[d^2] - E[d]^2 (unbiased=False). Computed in place into s2; the
    # one-pass form can go slightly negative from cancellation, so clamp for the
    # ranking (magnitude only affects which elements are 'most variable').
    s2.div_(n).sub_((s1 / n).square_()).clamp_(min=0)  # s2 is now var

    # 'Select': top-`select_topk` fraction of highest-variance elements.
    mask = None
    if select_topk < 1:
        mask = _sce_select_mask(s2, select_topk)
        if int(torch.count_nonzero(mask)) == 0:
            return torch.zeros(shape, dtype=dtype, device=device)
    del s2  # free variance buffer

    # Elected per-element sign (TIES 'sum' method). Masking is a shared per-element
    # factor, so sum_i(mask * d_i) == mask * s1 and the sign is unchanged where
    # mask > 0; masked-out elements get sign +1 but are excluded below anyway.
    if mask is not None:
        s1 *= mask
    majority_sign = (s1 >= 0).to(dtype) * 2 - 1
    del s1

    # Apply the 'select' mask in place -> task vectors.
    if mask is not None:
        for i in range(n):
            deltas[i] *= mask
        del mask

    if not normalize:
        # Additive convention: SUM the sign-agreeing selected contributions. No
        # per-tensor weighting, no divide -- strengths (already baked into the
        # deltas) act as gains. Accumulate in place; hold only {acc, majority_sign}.
        acc = None
        for i in range(n):
            tv = deltas[i]
            deltas[i] = None                   # free input as soon as consumed
            agree = (torch.sign(tv) == majority_sign).to(dtype)  # 0/1
            tv.mul_(agree)                     # keep only agreeing elements
            acc = tv if acc is None else acc.add_(tv)
        return acc

    # 'Calculate' (normalized average): per-tensor SCE weights = mean(tv_i**2),
    # normalized over i. sum(tv_i**2) == ||tv_i||^2, so use the norm reduction (no
    # full-layer square temporary).
    denom = float(deltas[0].numel())
    tv_w = torch.empty(n, dtype=torch.float32, device=device)
    for i in range(n):
        tv_w[i] = deltas[i].float().norm() ** 2 / denom
    wsum = float(tv_w.sum())
    tv_w = torch.ones_like(tv_w) / n if abs(wsum) < 1e-6 else tv_w / wsum

    # 'Erase' + merge: keep only contributions agreeing with the elected sign,
    # weighted by the SCE weights, then normalize by the surviving weight sum.
    # Accumulate in place to hold only {numerator, divisor, majority_sign} beyond
    # the shrinking delta list.
    numerator = None
    divisor = None
    for i in range(n):
        tv = deltas[i]
        deltas[i] = None                       # free input as soon as consumed
        w_i = float(tv_w[i])
        agree = (torch.sign(tv) == majority_sign).to(dtype)  # 0/1
        tv.mul_(agree).mul_(w_i)               # tv <- w_i * agree * tv (in place)
        numerator = tv if numerator is None else numerator.add_(tv)
        agree.mul_(w_i)                        # agree <- w_i * agree (in place)
        divisor = agree if divisor is None else divisor.add_(agree)
    return numerator / divisor.clamp_(min=1e-6)


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