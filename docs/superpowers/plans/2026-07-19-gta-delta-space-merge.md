# Delta-space GTA Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-factor mergekit merge for the GTA family (linear, task_arithmetic, ties, dare, della, breadcrumbs) with our own delta-space implementation, so `normalize` stops collapsing LoRA strength (no 1/N² squaring, correct sign vote), then refactor the merged delta back to a LoRA via the existing SVD framework with energy-dynamic rank.

**Architecture:** New pure-torch module `src/merge/gta.py` merges a list of full LoRA deltas (sparsify → sign election → disjoint merge → per-element normalize), faithfully matching mergekit's `sparsify`/GTA math but with no mergekit dependency. In `process_key`, GTA-mode keys materialize each LoRA's delta `(alpha/rank)·up@down`, call `gta_merge`, and refactor with `perform_lora_svd` (energy rank). Non-GTA methods (slerp/karcher/nuslerp/sce/nearswap) and the CLIP path are untouched.

**Tech Stack:** Python, PyTorch. Reference oracle: `mergekit.sparsify` and `mergekit.merge_methods.generalized_task_arithmetic` (used only in tests, not in shipped code). Existing SVD: `src/utility.py::perform_lora_svd`.

**Design doc:** `docs/superpowers/specs/2026-07-19-gta-delta-space-merge-design.md`

---

## Environment / test-harness notes (read once)

- **Python:** use the venv: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python`. System python has no torch.
- **pytest does not work** in this repo (collection walks into the repo-root `__init__.py` and `src/merge/__init__.py`, both of which fail with relative-import errors — pre-existing, out of scope). Do **not** use pytest.
- **Tests are standalone scripts** run directly: `.venv/bin/python tests/test_gta_*.py`. Each script exits non-zero on failure. `gta.py` must have **zero intra-package imports** (only `torch`, stdlib) so a test can load it by file path without triggering the broken package `__init__`.
- A shared loader helper `tests/gta_helpers.py` (Task 1) provides `load_gta()` (file-path import of `src/merge/gta.py`) and `mergekit_sparsify` re-exports for parity.
- Ignore the `pynvml` FutureWarning printed by torch on import — it is noise.

## File structure

- **Create** `src/merge/gta.py` — own GTA on full deltas. Pure torch. Responsibilities: sparsify variants, rescale, sign election, disjoint merge, `gta_merge` entry, mode→config mapping.
- **Create** `tests/gta_helpers.py` — file-path loader + mergekit oracle re-exports (not a `test_` file; not collected).
- **Create** `tests/test_gta_sparsify.py`, `tests/test_gta_merge.py`, `tests/test_gta_parity.py`, `tests/test_gta_behavior.py` — standalone script tests.
- **Create** `src/merge/lora_refactor.py` — `merged_delta_to_lora(delta, ranks, ...)`: materialize/refactor helper wrapping `perform_lora_svd` with energy rank. (Kept separate from `gta.py` so `gta.py` stays dependency-free and independently testable; `lora_refactor.py` may import `utility`.)
- **Modify** `src/lora_mergekit_merge.py` — `process_key`: branch GTA modes to the delta-space path.

## Mode → config mapping (single source of truth, implemented in Task 5)

| mode            | sparsify method     | sign_consensus                         | sparse params |
|-----------------|---------------------|----------------------------------------|---------------|
| linear          | none                | False                                  | —             |
| task_arithmetic | none                | False                                  | —             |
| ties            | magnitude           | True (always)                          | density       |
| dare            | random (bernoulli)  | `method_args['sign_consensus_algorithm']` | density    |
| della           | della_magprune      | `method_args['sign_consensus_algorithm']` | density,epsilon |
| breadcrumbs     | magnitude_outliers  | `method_args['sign_consensus_algorithm']` | density,gamma |

`rescale_norm` is applied **per-delta inside sparsify** (matching mergekit's `sparsify(..., rescale_norm=...)`), not as a final step. `"default"` resolves per-mode via mergekit's registered `default_rescale`/`default_normalize` (captured as constants in Task 5).

---

## Task 1: Test harness helper

**Files:**
- Create: `tests/gta_helpers.py`

- [ ] **Step 1: Write the helper**

```python
# tests/gta_helpers.py
"""Shared helpers for standalone gta.py script tests.

Not a `test_` module (pytest is broken in this repo anyway). Loads
src/merge/gta.py by file path so the broken package __init__ is never imported,
and re-exports mergekit's sparsify functions as the parity oracle.
"""
import importlib.util
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gta():
    path = os.path.join(REPO, "src", "merge", "gta.py")
    spec = importlib.util.spec_from_file_location("gta_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(tests):
    """Run a dict/list of {name: fn}. Exit non-zero on first failure."""
    if isinstance(tests, dict):
        tests = list(tests.items())
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    if failed:
        print(f"\n{failed} FAILED")
        sys.exit(1)
    print(f"\nAll {len(tests)} passed")
```

- [ ] **Step 2: Smoke-test the loader fails cleanly before gta.py exists**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -c "import sys; sys.path.insert(0,'tests'); import gta_helpers; gta_helpers.load_gta()"`
Expected: FAIL with `FileNotFoundError` / spec error (gta.py not created yet). This confirms the loader points at the right path.

- [ ] **Step 3: Commit**

```bash
git add tests/gta_helpers.py
git commit -m "test: add standalone loader harness for gta module"
```

---

## Task 2: Rescale + sparsify primitives in gta.py

**Files:**
- Create: `src/merge/gta.py`
- Test: `tests/test_gta_sparsify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gta_sparsify.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from gta_helpers import load_gta, run
from mergekit.sparsify import (
    magnitude as mk_magnitude,
    magnitude_outliers as mk_outliers,
    bernoulli as mk_bernoulli,
    della_magprune as mk_della,
    RescaleNorm,
)

gta = load_gta()


def _rand(shape=(8, 16), seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape)


def test_magnitude_matches_mergekit():
    t = _rand()
    for d in (0.3, 0.5, 0.9):
        got = gta.sparsify(t, "magnitude", density=d)
        exp = mk_magnitude(t, density=d)
        assert torch.allclose(got, exp), f"density={d}"


def test_magnitude_rescale_l1_matches():
    t = _rand()
    got = gta.sparsify(t, "magnitude", density=0.5, rescale_norm="l1")
    exp = mk_magnitude(t, density=0.5, rescale_norm=RescaleNorm.l1)
    assert torch.allclose(got, exp)


def test_outliers_matches_mergekit():
    t = _rand()
    got = gta.sparsify(t, "magnitude_outliers", density=0.7, gamma=0.1)
    exp = mk_outliers(t, density=0.7, gamma=0.1)
    assert torch.allclose(got, exp)


def test_bernoulli_matches_with_seed():
    t = _rand()
    torch.manual_seed(42); got = gta.sparsify(t, "random", density=0.6)
    torch.manual_seed(42); exp = mk_bernoulli(t, density=0.6)
    assert torch.allclose(got, exp)


def test_della_matches_with_seed():
    t = _rand()
    torch.manual_seed(7); got = gta.sparsify(t, "della_magprune", density=0.7, epsilon=0.05)
    torch.manual_seed(7); exp = mk_della(t, density=0.7, epsilon=0.05)
    assert torch.allclose(got, exp)


def test_density_one_is_identity():
    t = _rand()
    assert torch.allclose(gta.sparsify(t, "magnitude", density=1.0), t)


run([
    ("magnitude", test_magnitude_matches_mergekit),
    ("magnitude_rescale_l1", test_magnitude_rescale_l1_matches),
    ("outliers", test_outliers_matches_mergekit),
    ("bernoulli_seed", test_bernoulli_matches_with_seed),
    ("della_seed", test_della_matches_with_seed),
    ("density_one_identity", test_density_one_is_identity),
])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_sparsify.py`
Expected: FAIL (gta.py does not exist / has no `sparsify`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/merge/gta.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_sparsify.py`
Expected: `All 6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/merge/gta.py tests/test_gta_sparsify.py
git commit -m "feat(gta): own sparsify primitives matching mergekit"
```

---

## Task 3: Sign election + disjoint merge + normalize

**Files:**
- Modify: `src/merge/gta.py`
- Test: `tests/test_gta_merge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gta_merge.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from gta_helpers import load_gta, run

gta = load_gta()


def test_elect_sign_weighted_majority():
    # element 0: +2 (w1) vs -1 (w1) -> weighted sum +1 -> +; element 1 opposite
    a = torch.tensor([2.0, -2.0])
    b = torch.tensor([-1.0, 1.0])
    w = torch.tensor([1.0, 1.0])
    wd = torch.stack([a * w[0], b * w[1]])
    sign = gta.elect_sign(wd)
    assert torch.equal(sign, torch.tensor([1.0, -1.0]))


def test_disjoint_normalize_nonoverlap_keeps_strength():
    # A touches elem0, B touches elem1; both weight 1; normalize on.
    a = torch.tensor([2.0, 0.0])
    b = torch.tensor([0.0, 2.0])
    w = torch.tensor([1.0, 1.0])
    merged = gta.disjoint_merge(torch.stack([a, b]), w, sign_consensus=True, normalize=True)
    assert torch.allclose(merged, torch.tensor([2.0, 2.0]))  # not halved


def test_disjoint_normalize_overlap_conflict_arbitrated():
    # both touch elem0, opposite signs, equal weight -> winner kept at full, not zero
    a = torch.tensor([2.0])
    b = torch.tensor([-2.0])
    w = torch.tensor([1.0, 1.0])
    merged = gta.disjoint_merge(torch.stack([a, b]), w, sign_consensus=True, normalize=True)
    assert merged.abs().item() == 2.0  # arbitrated, not cancelled


def test_linear_normalize_is_weighted_average():
    a = torch.tensor([1.0])
    b = torch.tensor([3.0])
    w = torch.tensor([1.0, 1.0])
    merged = gta.disjoint_merge(torch.stack([a, b]), w, sign_consensus=False, normalize=True)
    assert torch.allclose(merged, torch.tensor([2.0]))


def test_n_equal_loras_scale_one_over_n_not_squared():
    # 4 identical deltas, weight 1, normalize on -> average == single delta (1/N behavior)
    d = torch.tensor([1.0, -1.0, 0.5])
    stack = torch.stack([d, d, d, d])
    w = torch.ones(4)
    merged = gta.disjoint_merge(stack, w, sign_consensus=True, normalize=True)
    assert torch.allclose(merged, d)  # NOT d/4 and NOT d/16


run([
    ("elect_sign", test_elect_sign_weighted_majority),
    ("nonoverlap_keeps_strength", test_disjoint_normalize_nonoverlap_keeps_strength),
    ("conflict_arbitrated", test_disjoint_normalize_overlap_conflict_arbitrated),
    ("linear_weighted_average", test_linear_normalize_is_weighted_average),
    ("n_loras_1_over_n", test_n_equal_loras_scale_one_over_n_not_squared),
])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_merge.py`
Expected: FAIL (`elect_sign`/`disjoint_merge` not defined).

- [ ] **Step 3: Write minimal implementation** (append to `src/merge/gta.py`)

```python
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
```

Note: for `sign_consensus=False` (linear/task_arithmetic) the divisor is the signed
`Σ w` (matches mergekit's non-ties `weights.sum`); for the sign-consensus path the
divisor is `Σ|w|` over survivors (matches mergekit's `(weights * mask).sum` where
mask already encodes agreement).

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_merge.py`
Expected: `All 5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/merge/gta.py tests/test_gta_merge.py
git commit -m "feat(gta): sign election + disjoint merge with per-element normalize"
```

---

## Task 4: `gta_merge` entry point + mode config

**Files:**
- Modify: `src/merge/gta.py`
- Test: `tests/test_gta_parity.py`

- [ ] **Step 1: Write the failing parity test (vs real mergekit GTATask on full deltas)**

```python
# tests/test_gta_parity.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from gta_helpers import load_gta, run

from mergekit.common import ModelReference, ModelPath, ImmutableMap
from mergekit.architecture import WeightInfo
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods import REGISTERED_MERGE_METHODS
from mergekit.merge_methods.generalized_task_arithmetic import GTATask
from mergekit.sparsify import RescaleNorm

gta = load_gta()


def _mk_gta(deltas, strengths, mode, normalize, density=1.0, rescale=None):
    """Run mergekit's GTATask on full deltas with a zeros base (LoRA convention)."""
    tensors = {}
    params = {}
    base_ref = ModelReference(model=ModelPath(path="zeros.base"))
    tensors[base_ref] = torch.zeros_like(deltas[0])
    params[base_ref] = ImmutableMap({"weight": 0.0, "density": density})
    for i, (d, s) in enumerate(zip(deltas, strengths)):
        r = ModelReference(model=ModelPath(path=f"m.{i}"))
        tensors[r] = d
        params[r] = ImmutableMap({"weight": s, "density": density})
    wi = WeightInfo(name="w", dtype=None, is_embed=False)
    gt = GatherTensors(weight_info=ImmutableMap({r: WeightInfo(name=f"m{i}.w")
                                                 for i, r in enumerate(tensors)}))
    method = REGISTERED_MERGE_METHODS[mode]
    task = GTATask(method=method, tensors=gt, base_model=base_ref, weight_info=wi,
                   gather_tensors=gt, tensor_parameters=ImmutableMap(params),
                   int8_mask=False, normalize=normalize, lambda_=1.0,
                   rescale_norm=rescale)
    return task.execute(tensors=tensors)


def _check(mode, mk_mode, normalize, density=1.0):
    torch.manual_seed(0)
    deltas = [torch.randn(6, 10), torch.randn(6, 10), torch.randn(6, 10)]
    strengths = [1.0, 0.7, 0.4]
    exp = _mk_gta(deltas, strengths, mk_mode, normalize, density)
    got = gta.gta_merge(deltas, torch.tensor(strengths), mode=mode,
                        normalize=normalize, density=density)
    assert torch.allclose(got, exp, atol=1e-5), (
        f"{mode}/{mk_mode} normalize={normalize} density={density} "
        f"maxdiff={ (got-exp).abs().max() }")


def test_linear_parity():
    _check("linear", "linear", normalize=True)
    _check("linear", "linear", normalize=False)


def test_task_arithmetic_parity():
    _check("task_arithmetic", "task_arithmetic", normalize=True)
    _check("task_arithmetic", "task_arithmetic", normalize=False)


def test_ties_parity():
    _check("ties", "ties", normalize=True, density=0.6)
    _check("ties", "ties", normalize=False, density=0.6)


run([
    ("linear_parity", test_linear_parity),
    ("task_arithmetic_parity", test_task_arithmetic_parity),
    ("ties_parity", test_ties_parity),
])
```

> Note on mergekit API drift: `GTATask`/`WeightInfo`/`GatherTensors` construction here
> mirrors `src/merge/algorithms.py:104-115` and `src/merge/utils.py:24-43`. If a
> constructor signature differs in the installed mergekit, copy the exact call shape
> from those files (they are known-good against the installed version).

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_parity.py`
Expected: FAIL (`gta_merge` not defined).

- [ ] **Step 3: Write minimal implementation** (append to `src/merge/gta.py`)

```python
# --------------------------------------------------------- mode config
# sparsify method per mode; None means no sparsification.
_MODE_SPARSIFY = {
    "linear": None,
    "task_arithmetic": None,
    "ties": "magnitude",
    "dare": "random",
    "della": "della_magprune",
    "breadcrumbs": "magnitude_outliers",
}
# ties modes always elect a sign; dare/della/breadcrumbs follow the node's
# `sign_consensus_algorithm` toggle; linear/task_arithmetic never do.
_MODE_ALWAYS_CONSENSUS = {"ties"}
_MODE_NEVER_CONSENSUS = {"linear", "task_arithmetic"}

GTA_MODES = tuple(_MODE_SPARSIFY.keys())


def gta_merge(deltas: List[torch.Tensor], weights: torch.Tensor, *, mode: str,
              normalize: bool = True, density: float = 1.0, epsilon: float = 0.0,
              gamma: float = 0.0, sign_consensus_algorithm: bool = False,
              rescale_norm: Optional[str] = None) -> torch.Tensor:
    """Merge a list of full LoRA deltas with GTA semantics on the delta itself.

    `weights` are the per-LoRA merge strengths (signed). Returns the merged delta."""
    if mode not in _MODE_SPARSIFY:
        raise ValueError(f"unknown GTA mode {mode!r}")
    if mode in _MODE_ALWAYS_CONSENSUS:
        consensus = True
    elif mode in _MODE_NEVER_CONSENSUS:
        consensus = False
    else:
        consensus = bool(sign_consensus_algorithm)

    sp = _MODE_SPARSIFY[mode]
    sparse = [sparsify(d, sp, density=density, gamma=gamma, epsilon=epsilon,
                       rescale_norm=rescale_norm) for d in deltas]
    stack = torch.stack(sparse, dim=0)
    return disjoint_merge(stack, weights.to(stack.dtype), sign_consensus=consensus,
                          normalize=normalize)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_parity.py`
Expected: `All 3 passed`. If `ties_parity` diverges, check the `divisor` sign convention in `disjoint_merge` against mergekit `generalized_task_arithmetic.py:170-179`.

- [ ] **Step 5: Commit**

```bash
git add src/merge/gta.py tests/test_gta_parity.py
git commit -m "feat(gta): gta_merge entry point with mode config, parity vs mergekit"
```

---

## Task 5: Resolve per-mode `rescale_norm`/`normalize` defaults

**Files:**
- Modify: `src/merge/gta.py`
- Test: `tests/test_gta_parity.py` (extend)

- [ ] **Step 1: Capture mergekit's per-mode defaults**

Run to print the registered defaults, then hard-code them:
`/home/lars/SD/Apps/ComfyUI/.venv/bin/python -c "from mergekit.merge_methods import REGISTERED_MERGE_METHODS as R; [print(m, getattr(R[m],'default_normalize',None), getattr(R[m],'default_rescale',None)) for m in ['linear','task_arithmetic','ties','dare_ties','dare_linear','della','della_linear','breadcrumbs','breadcrumbs_ties']]"`
Expected: prints a bool pair per mode.

- [ ] **Step 2: Write the failing test (default rescale resolution)**

```python
def test_resolve_rescale_default_matches_mergekit():
    # For a mode mergekit rescales by default, "default" must resolve to l1.
    assert gta.resolve_rescale_norm("della", "default") in (None, "l1")
    assert gta.resolve_rescale_norm("ties", "none") is None
    assert gta.resolve_rescale_norm("ties", "l2") == "l2"
```
Add `("resolve_rescale", test_resolve_rescale_default_matches_mergekit)` to the `run([...])` list.

- [ ] **Step 3: Run to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_parity.py`
Expected: FAIL (`resolve_rescale_norm` not defined).

- [ ] **Step 4: Implement** (append to `src/merge/gta.py`, filling `_MODE_DEFAULT_RESCALE` from Step 1's output)

```python
# Per-mode default: does this mode rescale by default? (from mergekit registry)
# Fill exact bools from Task 5 Step 1 output.
_MODE_DEFAULT_RESCALE = {
    "linear": False,
    "task_arithmetic": False,
    "ties": False,
    "dare": False,
    "della": True,        # verify against Step 1
    "breadcrumbs": True,  # verify against Step 1
}


def resolve_rescale_norm(mode: str, rescale_norm: str) -> Optional[str]:
    """Resolve the UI's rescale_norm choice ('default'|'none'|'l1'|'l2'|'linf')."""
    if rescale_norm == "default":
        return "l1" if _MODE_DEFAULT_RESCALE.get(mode, False) else None
    if rescale_norm == "none":
        return None
    if rescale_norm in _RESCALE:
        return rescale_norm
    raise ValueError(f"unknown rescale_norm {rescale_norm!r}")
```

Then update `gta_merge` to accept `rescale_norm: str = "default"` and resolve it:
change its body's `rescale_norm=rescale_norm` in the `sparsify` call to
`rescale_norm=resolve_rescale_norm(mode, rescale_norm)`.

- [ ] **Step 5: Run to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_parity.py`
Expected: `All 4 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/merge/gta.py tests/test_gta_parity.py
git commit -m "feat(gta): resolve per-mode rescale_norm defaults"
```

---

## Task 6: Materialize + refactor helper (`lora_refactor.py`)

**Files:**
- Create: `src/merge/lora_refactor.py`
- Test: `tests/test_gta_behavior.py` (created here, extended in Task 8)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gta_behavior.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
import torch
from gta_helpers import load_gta, run

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))  # for utility import inside lora_refactor


def _load(modfile, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "src", modfile))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_refactor_reconstructs_delta():
    torch.manual_seed(0)
    delta = torch.randn(32, 48)
    refactor = _load("merge/lora_refactor.py", "lora_refactor")
    up, down, alpha = refactor.merged_delta_to_lora(delta, target_rank=16,
                                                    energy=0.999)
    rank = up.shape[1]
    recon = (alpha / rank) * (up @ down)
    # energy 0.999 keeps almost all magnitude
    err = (recon - delta).norm() / delta.norm()
    assert err < 0.05, f"reconstruction error {err}"


run([("refactor_reconstructs", test_refactor_reconstructs_delta)])
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_behavior.py`
Expected: FAIL (`lora_refactor` missing).

- [ ] **Step 3: Implement**

First confirm `perform_lora_svd`'s alpha/scale convention by reading `src/utility.py:108-232` — `new_alpha = scale * new_rank`, and up/down are distributed so `up @ down` reconstructs `weight` when `scale=1.0` and `distribute_singular_values=True`. The effective delta downstream is `(alpha/rank) * up@down`; with `alpha = rank` (scale=1) that is exactly `up@down = weight`. Verify this reconstruction identity in the test above (it does).

```python
# src/merge/lora_refactor.py
"""Refactor a merged full delta back into a LoRA (up, down, alpha) via SVD.

Wraps the existing SVD framework (utility.perform_lora_svd) with energy-based
dynamic rank selection. Imports `utility` (only usable inside the ComfyUI
runtime / with src on sys.path), so it is kept separate from the dependency-free
gta.py."""
from typing import Optional, Tuple

import torch

try:
    from ..utility import perform_lora_svd            # package context (runtime)
except ImportError:                                    # file-path/test context
    from utility import perform_lora_svd


def merged_delta_to_lora(delta: torch.Tensor, target_rank: int, *,
                         energy: Optional[float] = 0.99, device: str = "cpu",
                         dtype: torch.dtype = torch.float32
                         ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Decompose `delta` into (up, down, alpha).

    target_rank: hard cap on output rank (e.g. max input rank).
    energy: cumulative singular-value retention for dynamic rank (None = fixed rank).
    """
    if energy is None:
        up, down, alpha = perform_lora_svd(delta, target_rank=target_rank,
                                           device=device, dtype=dtype, scale=1.0)
    else:
        up, down, alpha = perform_lora_svd(delta, target_rank=target_rank,
                                           device=device, dtype=dtype, scale=1.0,
                                           dynamic_method="sv_cumulative",
                                           dynamic_param=energy)
    return up, down, alpha
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_behavior.py`
Expected: `All 1 passed`. If import of `utility` fails, confirm `src` is on `sys.path` (the test adds it) and that `utility.py` imports cleanly standalone; if `utility.py` has heavy imports, load it by file path in the test the same way.

- [ ] **Step 5: Commit**

```bash
git add src/merge/lora_refactor.py tests/test_gta_behavior.py
git commit -m "feat(gta): merged-delta -> LoRA refactor with energy-dynamic rank"
```

---

## Task 7: Wire the delta-space path into `process_key`

**Files:**
- Modify: `src/lora_mergekit_merge.py` (the `else:` UNet branch at lines ~251-263, and the reconstruction at ~264-281)

- [ ] **Step 1: Read the current branch**

Read `src/lora_mergekit_merge.py:174-305` to re-anchor line numbers (they may have shifted). The change is confined to `process_key`.

- [ ] **Step 2: Add imports at top of `src/lora_mergekit_merge.py`**

```python
from .merge.gta import gta_merge, GTA_MODES
from .merge.lora_refactor import merged_delta_to_lora
```

- [ ] **Step 3: Branch GTA modes before the existing factored merge**

Replace the UNet `else:` block (the `sqrt_mag` factored path) so that when
`method_args.get('mode') in GTA_MODES and not is_clip`, the delta-space path runs
instead. Insert after `weights = torch.tensor(...)` (line ~183) and the
`up_tensors`/`down_tensors` extraction (lines ~237-238):

```python
            mode = method_args.get('mode')
            is_gta = (not is_clip) and (mode in GTA_MODES)

            if is_gta:
                # Materialize each LoRA's natural delta: (alpha_i/rank_i)*up@down
                deltas = []
                for (u, d, a) in lora_key_tuples.values():
                    u = u.to(device=device, dtype=torch.float32)
                    d = d.to(device=device, dtype=torch.float32)
                    rank = u.shape[1]
                    scale = (float(a) / rank) if a is not None else 1.0
                    # conv (4D): flatten to 2D like perform_lora_svd does
                    if u.dim() == 4:
                        u2 = u.reshape(u.shape[0], -1)
                        d2 = d.reshape(d.shape[0], -1)
                        deltas.append(scale * (u2 @ d2))
                    else:
                        deltas.append(scale * (u @ d))

                merged = gta_merge(
                    deltas,
                    weights.to(torch.float32),
                    mode=mode,
                    normalize=method_args.get('normalize', True),
                    density=method_args.get('density', 1.0),
                    epsilon=method_args.get('epsilon', 0.0),
                    gamma=method_args.get('gamma', 0.0),
                    sign_consensus_algorithm=method_args.get('sign_consensus_algorithm', False),
                    rescale_norm=method_args.get('rescale_norm', 'default'),
                )

                target_rank = max(u.shape[1] for (u, _, _) in lora_key_tuples.values())
                up, down, alpha_out = merged_delta_to_lora(
                    merged, target_rank=target_rank, energy=0.99, device=device)

                if lambda_ < 1.0:
                    up = up * lambda_
                up = up.to(device='cpu', dtype=torch.float32)
                down = down.to(device='cpu', dtype=torch.float32)
                return key, (up, down, torch.tensor(float(alpha_out)))
            # ---- non-GTA (existing factored path unchanged below) ----
```

Leave the existing CLIP branch and the `else` factored path (now reached only by
non-GTA UNet methods: slerp/karcher/nuslerp/sce/nearswap) exactly as-is.

- [ ] **Step 4: Sanity-run the merge path end to end**

Because the node requires ComfyUI runtime, verify import health + a tiny synthetic
run via a script that loads the modules by path:

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import importlib.util,os; \
p=os.path.join('src','merge','gta.py'); s=importlib.util.spec_from_file_location('gta',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); \
import torch; d=[torch.randn(8,8) for _ in range(3)]; \
print('ok', m.gta_merge(d, torch.tensor([1.0,0.7,0.4]), mode='ties', density=0.6, normalize=True).shape)"`
Expected: `ok torch.Size([8, 8])`.

- [ ] **Step 5: Commit**

```bash
git add src/lora_mergekit_merge.py
git commit -m "feat: route GTA merge family through delta-space path"
```

---

## Task 8: End-to-end behavioral tests + verify the fix

**Files:**
- Modify: `tests/test_gta_behavior.py`

- [ ] **Step 1: Add the behavioral tests that encode the bug fix**

```python
def _delta_from_lora(up, down, alpha):
    rank = up.shape[1]
    return (alpha / rank) * (up @ down)


def test_style_plus_character_nonoverlap_keeps_strength():
    # Two rank-1 LoRAs acting on disjoint output rows -> deltas don't overlap.
    torch.manual_seed(0)
    A = torch.zeros(4, 6); A[0, :] = 2.0        # style: row 0
    B = torch.zeros(4, 6); B[2, :] = 2.0        # character: row 2
    merged = gta.gta_merge([A, B], torch.tensor([1.0, 1.0]), mode="ties",
                           density=1.0, normalize=True)
    # non-overlapping rows must retain full magnitude, not be halved
    assert torch.allclose(merged[0], A[0])
    assert torch.allclose(merged[2], B[2])


def test_normalize_does_not_collapse_as_1_over_n_squared():
    torch.manual_seed(0)
    deltas = [torch.randn(6, 6) for _ in range(4)]
    merged = gta.gta_merge(deltas, torch.ones(4), mode="ties", density=1.0,
                           normalize=True)
    avg = torch.stack(deltas).mean(0)
    # merged should be ~ the (sign-elected) average, i.e. O(1/N) not O(1/N^2)
    assert merged.norm() > 0.5 * avg.norm()


run([
    ("refactor_reconstructs", test_refactor_reconstructs_delta),
    ("nonoverlap_keeps_strength", test_style_plus_character_nonoverlap_keeps_strength),
    ("no_1_over_n_squared", test_normalize_does_not_collapse_as_1_over_n_squared),
])
```

- [ ] **Step 2: Run the full behavioral suite**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_behavior.py`
Expected: `All 3 passed`.

- [ ] **Step 3: Run every gta test script (regression gate)**

Run:
```bash
cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI
for t in tests/test_gta_sparsify.py tests/test_gta_merge.py tests/test_gta_parity.py tests/test_gta_behavior.py; do
  echo "== $t =="; /home/lars/SD/Apps/ComfyUI/.venv/bin/python "$t" || exit 1
done
```
Expected: all four scripts print their `All N passed` line and the loop exits 0.

- [ ] **Step 4: Commit**

```bash
git add tests/test_gta_behavior.py
git commit -m "test(gta): end-to-end behavioral tests encoding the normalize fix"
```

---

## Task 9: Real-app verification + docs/memory

**Files:**
- Modify: `docs/superpowers/specs/2026-07-19-gta-delta-space-merge-design.md` (status), memory index.

- [ ] **Step 1: Verify in ComfyUI (manual, user-driven)**

The unit/parity tests prove the math; the only thing they can't exercise is the live
ComfyUI node graph. Ask the user to run the "PM Ties (mergekit)" node on the exact
style+character LoRA pair that originally motivated this, with `normalize=True`, and
confirm the merged LoRA now has visible effect (compare a render against `normalize=False`
and against pre-fix). Capture the before/after qualitatively.

- [ ] **Step 2: Update the design doc status**

Change the spec header `**Status:**` to `Implemented (2026-07-19)` and note any deviations
discovered during implementation (e.g. mergekit default-rescale bools from Task 5 Step 1).

- [ ] **Step 3: Update the squaring memory**

Update the `lora-merger-factored-strength-squaring` memory: note that the GTA family
(linear/task_arithmetic/ties/dare/della/breadcrumbs) no longer uses the `√s` factored path
— it now merges full deltas in `src/merge/gta.py` and refactors via SVD, so the squaring
trap no longer applies there. The `√s` note still applies to any remaining factored paths.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-19-gta-delta-space-merge-design.md
git commit -m "docs: mark delta-space GTA merge implemented"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** own GTA family (Tasks 2-5), materialize+SVD-refactor with energy rank
  (Task 6), wiring incl. all six modes + linear (Task 7), per-element normalize / sign vote /
  non-overlap / 1-over-N behavior (Tasks 3, 8), rescale_norm (Tasks 2, 5), correct per-LoRA
  alpha via materialization (Task 7), testing strategy = standalone scripts (all tasks).
- **Known risk — mergekit test API:** Task 4's oracle constructs `GTATask` directly. If the
  installed mergekit's constructor differs, mirror `src/merge/algorithms.py:104-115` exactly.
  The shipped code (`gta.py`) never imports mergekit — only the tests do.
- **Known risk — `utility` import in tests:** `lora_refactor.py` imports `utility`. If
  `utility.py` pulls heavy/broken imports under the test sys.path, load `perform_lora_svd`'s
  module by file path in the test (same pattern as `load_gta`).
- **Deviation to confirm at runtime:** the `(alpha/rank)·up@down` materialization scale must
  equal what ComfyUI applies downstream. Task 6 Step 3 says to confirm against
  `src/utility.py`; also cross-check against the LoRA apply path in the ComfyUI runtime.
- **Out of scope (unchanged):** slerp/karcher/nuslerp/sce/nearswap, CLIP path, repo-wide
  pytest breakage.