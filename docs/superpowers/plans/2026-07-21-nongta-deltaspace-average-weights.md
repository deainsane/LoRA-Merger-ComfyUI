# Delta-space + `average_weights` for SLERP/NuSLERP/Karcher/NearSwap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move SLERP, NuSLERP, Karcher, NearSwap onto the delta-space merge path and give each an `average_weights` toggle with the same meaning as every other merge node.

**Architecture:** A new `interp_delta_merge` helper runs the existing mergekit method function on full per-LoRA deltas with unit weights to get a blend `B` (‖B‖≈one delta), then post-scales by the per-LoRA strengths (Σ|s| for additive/OFF, mean(|s|) for average/ON). The merger node routes these four modes through that helper inside the existing delta-space branch, then refactors the merged delta back into a LoRA via `merged_delta_to_lora`. Four ComfyUI nodes gain an `average_weights` boolean wired to the internal `normalize` setting.

**Tech Stack:** Python, PyTorch, mergekit, ComfyUI custom node.

**Repo test reality:** pytest does NOT collect in this repo (known import-path breakage). Tests are standalone scripts run with `.venv/bin/python <file>` from the ComfyUI root (`/home/lars/SD/Apps/ComfyUI`). They use a small `run([...])` harness and print `PASS`/`FAIL`. Follow that pattern; do not add pytest.

**Design doc:** `docs/superpowers/specs/2026-07-21-nongta-deltaspace-average-weights-design.md`

---

## File Structure

- `src/merge/algorithms.py` — add `INTERP_MODES` constant and `interp_delta_merge()` helper (co-located with `sce_merge_deltas`).
- `src/merge/__init__.py` — export `interp_delta_merge`, `INTERP_MODES`.
- `src/lora_mergekit_merge.py` — route the four modes through `interp_delta_merge` in the delta-space branch; add the modes to the CUDA worker-serialization predicate.
- `src/nodes_merge_methods.py` — add `average_weights` widget + `normalize` setting to the four nodes.
- `tests/test_interp_delta_merge.py` — NEW standalone test for the helper.
- `tests/test_merge_node_names.py` — extend with the four nodes' `average_weights`→`normalize` wiring.

All commands below are run from `/home/lars/SD/Apps/ComfyUI` (the ComfyUI root). The custom node lives at `custom_nodes/LoRA-Merger-ComfyUI`.

---

## Task 1: `interp_delta_merge` helper + `INTERP_MODES`

**Files:**
- Modify: `custom_nodes/LoRA-Merger-ComfyUI/src/merge/algorithms.py`
- Modify: `custom_nodes/LoRA-Merger-ComfyUI/src/merge/__init__.py`
- Test: `custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_delta_merge.py` (new)

- [ ] **Step 1: Write the failing test**

Create `custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_delta_merge.py`:

```python
# tests/test_interp_delta_merge.py
# Standalone script test (repo pytest is broken). Verifies the delta-space
# interpolation merge helper: unit-weight blend + strength post-scale.
import os, sys, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)            # .../custom_nodes
COMFY_ROOT = os.path.dirname(PARENT)      # ComfyUI root, so `import comfy` resolves
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, PARENT)

import torch
from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, ModelPath, ImmutableMap
from mergekit.io.tasks import GatherTensors

# import the package under a synthetic name so relative imports resolve
import importlib.util
PKG = "LoRA_Merger_ComfyUI_test"
spec = importlib.util.spec_from_file_location(
    PKG, os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
pkg = importlib.util.module_from_spec(spec)
sys.modules[PKG] = pkg
spec.loader.exec_module(pkg)

from LoRA_Merger_ComfyUI_test.src.merge.algorithms import (
    interp_delta_merge, INTERP_MODES,
    slerp_merge, nuslerp_merge, karcher_merge, nearswap_merge,
)
from LoRA_Merger_ComfyUI_test.src.merge.utils import create_map, create_tensor_param

METHODS = {
    "slerp":   (slerp_merge,   {"mode": "slerp", "t": 0.5, "lambda_": 1.0}),
    "nuslerp": (nuslerp_merge, {"mode": "nuslerp", "nuslerp_flatten": True,
                                "nuslerp_row_wise": False, "lambda_": 1.0}),
    "karcher": (karcher_merge, {"mode": "karcher", "max_iter": 10, "tol": 1e-5,
                                "lambda_": 1.0}),
    "nearswap": (nearswap_merge, {"mode": "nearswap", "similarity_threshold": 0.001,
                                  "lambda_": 1.0}),
}


def _delta(seed):
    torch.manual_seed(seed)
    return (torch.randn(64, 8) * 0.1) @ (torch.randn(8, 96) * 0.1)


def _blend_unit(method, deltas, margs):
    """Reference blend computed directly with unit weights (mirrors the helper)."""
    tm, wm = {}, {}
    wi = WeightInfo(name="k.merge", dtype=torch.float32, is_embed=False)
    for i, d in enumerate(deltas):
        ref = ModelReference(model=ModelPath(path=f"k.{i}"))
        tm[ref] = d
        wm[ref] = torch.tensor(1.0)
    gt = GatherTensors(weight_info=create_map("k", tm, torch.float32))
    tp = ImmutableMap({r: ImmutableMap(create_tensor_param(wm[r], margs)) for r in tm})
    return method(tm, gt, wi, tp, margs)


def test_modes_constant():
    assert INTERP_MODES == ("slerp", "nuslerp", "karcher", "nearswap"), INTERP_MODES


def test_additive_is_blend_times_sum():
    for name, (fn, margs) in METHODS.items():
        deltas = [_delta(1), _delta(2)]
        B = _blend_unit(fn, [d.clone() for d in deltas], dict(margs))
        w = torch.tensor([1.0, 0.5])
        out = interp_delta_merge(fn, [d.clone() for d in deltas], w, dict(margs),
                                 key="k", normalize=False)
        exp = B * float(w.abs().sum())          # additive -> * sum(|s|)
        assert torch.allclose(out, exp, atol=1e-5), f"{name}: additive != B*sum"


def test_average_is_blend_times_mean():
    for name, (fn, margs) in METHODS.items():
        deltas = [_delta(3), _delta(4)]
        B = _blend_unit(fn, [d.clone() for d in deltas], dict(margs))
        w = torch.tensor([1.0, 0.5])
        out = interp_delta_merge(fn, [d.clone() for d in deltas], w, dict(margs),
                                 key="k", normalize=True)
        exp = B * float(w.abs().mean())         # average -> * mean(|s|)
        assert torch.allclose(out, exp, atol=1e-5), f"{name}: average != B*mean"


def test_strength_is_linear_gain_additive():
    fn, margs = METHODS["slerp"]
    deltas = [_delta(5), _delta(6)]
    o1 = interp_delta_merge(fn, [d.clone() for d in deltas], torch.tensor([1.0, 1.0]),
                            dict(margs), key="k", normalize=False)
    o2 = interp_delta_merge(fn, [d.clone() for d in deltas], torch.tensor([2.0, 2.0]),
                            dict(margs), key="k", normalize=False)
    # sum(|s|) doubles from 2 -> 4, so magnitude doubles
    assert abs(o2.norm().item() / o1.norm().item() - 2.0) < 1e-4


def test_single_owner_fallback_not_zero():
    fn, margs = METHODS["slerp"]
    d = _delta(7)
    out = interp_delta_merge(fn, [d.clone()], torch.tensor([1.0]), dict(margs),
                             key="k", normalize=False)
    assert torch.allclose(out, d, atol=1e-6), "single-owner additive != delta*|s|"
    out2 = interp_delta_merge(fn, [d.clone()], torch.tensor([0.5]), dict(margs),
                              key="k", normalize=False)
    assert torch.allclose(out2, d * 0.5, atol=1e-6)


def run(tests):
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception:
            failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    if failed:
        print(f"\n{failed} FAILED"); sys.exit(1)
    print(f"\nAll {len(tests)} passed")


run([
    ("modes_constant", test_modes_constant),
    ("additive_is_blend_times_sum", test_additive_is_blend_times_sum),
    ("average_is_blend_times_mean", test_average_is_blend_times_mean),
    ("strength_linear_gain_additive", test_strength_is_linear_gain_additive),
    ("single_owner_fallback", test_single_owner_fallback_not_zero),
])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_delta_merge.py`
Expected: FAIL — `ImportError: cannot import name 'interp_delta_merge'` (helper not defined yet).

- [ ] **Step 3: Add the helper + constant to `algorithms.py`**

In `custom_nodes/LoRA-Merger-ComfyUI/src/merge/algorithms.py`, extend the utils import (it currently reads `from .utils import apply_weights_to_tensors`) to:

```python
from .utils import apply_weights_to_tensors, create_map, create_tensor_param
```

Then add the following immediately AFTER the `sce_merge_deltas` function definition ends (in the current file `sce_merge_deltas` is followed by `def karcher_merge`; insert this new code in between them):

```python
# Interpolation / averaging methods that merge in delta space via a unit-weight
# blend + a strength post-scale (see interp_delta_merge).
INTERP_MODES = ("slerp", "nuslerp", "karcher", "nearswap")


def interp_delta_merge(
    method,
    deltas: list,
    weights: torch.Tensor,
    method_args: Dict[str, Any],
    key: str = "merge",
    *,
    normalize: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Delta-space merge for the interpolation methods (slerp/nuslerp/karcher/
    nearswap).

    Runs ``method`` on the full per-LoRA ``deltas`` with UNIT weights to get the
    blend ``B`` (magnitude ~ one delta; these methods normalize/interpolate rather
    than sum). Then post-scales by the per-LoRA strengths:

      * ``normalize=False`` (additive, default): ``B * sum(|weights|)`` -- strengths
        act as gains, stacked LoRAs keep full magnitude.
      * ``normalize=True`` (average): ``B * mean(|weights|)`` -- a blend.

    Because the blend uses unit weights, relative strengths control MAGNITUDE only,
    not the interpolation position (that comes from each method's own parameter:
    slerp `t`, nearswap threshold, nuslerp/karcher default even).

    ``deltas`` are the alpha-scaled, strength-free per-LoRA deltas. ``weights`` is a
    1-D tensor of per-LoRA strengths aligned with ``deltas``.
    """
    if not deltas:
        raise ValueError("interp_delta_merge requires at least one delta")

    mag = weights.abs().to(torch.float32)
    strength_scale = float(mag.sum()) if not normalize else float(mag.mean())

    # A single contributing LoRA: the interpolation methods need >= 2 tensors, so
    # just scale the lone delta (sum and mean coincide for one element).
    if len(deltas) == 1:
        return deltas[0] * strength_scale

    tensor_map, weight_map = {}, {}
    weight_info = WeightInfo(name=f"{key}.merge", dtype=dtype, is_embed=False)
    for i, d in enumerate(deltas):
        ref = ModelReference(model=ModelPath(path=f"{key}.{i}"))
        tensor_map[ref] = d
        weight_map[ref] = torch.tensor(1.0)   # unit weight -> pure blend direction
    gather = GatherTensors(weight_info=create_map(key, tensor_map, dtype))
    params = ImmutableMap(
        {r: ImmutableMap(create_tensor_param(weight_map[r], method_args))
         for r in tensor_map})
    blend = method(tensor_map, gather, weight_info, params, method_args)
    return blend * strength_scale
```

Note: `WeightInfo`, `ModelReference`, `ModelPath`, `ImmutableMap`, `GatherTensors`, `Dict`, `Any`, and `torch` are already imported at the top of `algorithms.py`. Only `create_map`/`create_tensor_param` need adding to the utils import.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_delta_merge.py`
Expected: `All 5 passed`.

- [ ] **Step 5: Export from the merge package**

In `custom_nodes/LoRA-Merger-ComfyUI/src/merge/__init__.py`, change:

```python
from .algorithms import MERGE_ALGORITHMS, get_merge_algorithm, sce_merge_deltas
```

to:

```python
from .algorithms import (
    MERGE_ALGORITHMS, get_merge_algorithm, sce_merge_deltas,
    interp_delta_merge, INTERP_MODES,
)
```

and add `'interp_delta_merge',` and `'INTERP_MODES',` to the `__all__` list (next to `'sce_merge_deltas',`).

- [ ] **Step 6: Verify the package still imports**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'custom_nodes/LoRA-Merger-ComfyUI'); from src.merge import interp_delta_merge, INTERP_MODES; print(INTERP_MODES)"`
Expected: prints `('slerp', 'nuslerp', 'karcher', 'nearswap')` (ignore the pynvml FutureWarning).

- [ ] **Step 7: Commit**

```bash
git add custom_nodes/LoRA-Merger-ComfyUI/src/merge/algorithms.py \
        custom_nodes/LoRA-Merger-ComfyUI/src/merge/__init__.py \
        custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_delta_merge.py
git commit -m "feat(merge): interp_delta_merge helper for delta-space slerp/nuslerp/karcher/nearswap"
```

---

## Task 2: `average_weights` widget on the four nodes

**Files:**
- Modify: `custom_nodes/LoRA-Merger-ComfyUI/src/nodes_merge_methods.py`
- Test: `custom_nodes/LoRA-Merger-ComfyUI/tests/test_merge_node_names.py`

- [ ] **Step 1: Write the failing test**

In `custom_nodes/LoRA-Merger-ComfyUI/tests/test_merge_node_names.py`, add this function just before `def test_merger_output_scale():`:

```python
def test_interp_nodes_average_weights_wiring():
    # slerp/nuslerp/karcher/nearswap expose average_weights (default OFF) wired to
    # the internal "normalize" setting, like SCE and the GTA family.
    cases = {
        "SLERPMergeMethod": dict(),
        "NuSlerpMergeMethod": dict(),
        "KArcherMergeMethod": dict(),
        "NearSwapMergeMethod": dict(),
    }
    for cls_name in cases:
        cls = getattr(N, cls_name)
        req = cls.INPUT_TYPES()["required"]
        assert "average_weights" in req, f"{cls_name}: no average_weights widget"
        assert req["average_weights"][1]["default"] is False, \
            f"{cls_name}: average_weights default != False"
        s_off = cls().get_method(average_weights=False)[0]["settings"]
        assert s_off["normalize"] is False, f"{cls_name}: normalize not wired ({s_off})"
        s_on = cls().get_method(average_weights=True)[0]["settings"]
        assert s_on["normalize"] is True, f"{cls_name}: normalize ON not wired"
```

Also register it in the `run([...])` list (add before the `("merger_output_scale", ...)` line):

```python
        ("interp_nodes_average_weights_wiring", test_interp_nodes_average_weights_wiring),
```

And update the final success line `print(f"\nAll 5 passed")` to `print(f"\nAll 6 passed")`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_merge_node_names.py`
Expected: `FAIL interp_nodes_average_weights_wiring` with `AssertionError: SLERPMergeMethod: no average_weights widget`.

- [ ] **Step 3: Add `average_weights` to the four nodes**

In `custom_nodes/LoRA-Merger-ComfyUI/src/nodes_merge_methods.py`:

The reusable tooltip (use verbatim for all four):
```
"OFF: additive SUM, so per-LoRA strengths act as gains and stacked LoRAs keep full "
"magnitude (matches ComfyUI's native LoRA stacking and the other merge nodes, the "
"default). ON: normalized weighted AVERAGE, so strengths act as ratios and the result "
"is a blend/interpolation (weaker magnitude). Note: strengths control MAGNITUDE only, "
"not the interpolation position."
```

**(a) SLERPMergeMethod** — replace its `INPUT_TYPES` `required` dict and `get_method`:

```python
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "interpolation factor. At t=0 will return base_model, at t=1 will return the other one.",
                }),
                "average_weights": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF: additive SUM, so per-LoRA strengths act as gains and stacked LoRAs keep full "
                               "magnitude (matches ComfyUI's native LoRA stacking and the other merge nodes, the "
                               "default). ON: normalized weighted AVERAGE, so strengths act as ratios and the result "
                               "is a blend/interpolation (weaker magnitude). Note: strengths control MAGNITUDE only, "
                               "not the interpolation position.",
                }),
            },
        }
```

and

```python
    def get_method(self, t: float = 1., average_weights: bool = False):
        method_def = {
            "name": "slerp",
            "settings": {
                "t": t,
                "normalize": average_weights,
            }
        }
        return (method_def,)
```

**(b) NuSlerpMergeMethod** — add `average_weights` to `INPUT_TYPES` `required` (after `nuslerp_row_wise`) and update `get_method`:

```python
                "average_weights": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF: additive SUM, so per-LoRA strengths act as gains and stacked LoRAs keep full "
                               "magnitude (matches ComfyUI's native LoRA stacking and the other merge nodes, the "
                               "default). ON: normalized weighted AVERAGE, so strengths act as ratios and the result "
                               "is a blend/interpolation (weaker magnitude). Note: strengths control MAGNITUDE only, "
                               "not the interpolation position.",
                }),
```

```python
    def get_method(self, nuslerp_flatten: bool = True, nuslerp_row_wise: bool = False,
                   average_weights: bool = False):
        method_def = {
            "name": "nuslerp",
            "settings": {
                "nuslerp_flatten": nuslerp_flatten,
                "nuslerp_row_wise": nuslerp_row_wise,
                "normalize": average_weights,
            }
        }
        return (method_def,)
```

**(c) KArcherMergeMethod** — add `average_weights` to `INPUT_TYPES` `required` (after `tol`) and update `get_method`:

```python
                "average_weights": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF: additive SUM, so per-LoRA strengths act as gains and stacked LoRAs keep full "
                               "magnitude (matches ComfyUI's native LoRA stacking and the other merge nodes, the "
                               "default). ON: normalized weighted AVERAGE, so strengths act as ratios and the result "
                               "is a blend/interpolation (weaker magnitude). Note: strengths control MAGNITUDE only, "
                               "not the interpolation position.",
                }),
```

```python
    def get_method(self, max_iter: int = 10, tol: float = 0.5, average_weights: bool = False):
        method_def = {
            "name": "karcher",
            "settings": {
                "max_iter": max_iter,
                "tol": tol,
                "normalize": average_weights,
            }
        }
        return (method_def,)
```

**(d) NearSwapMergeMethod** — add `average_weights` to `INPUT_TYPES` `required` (after `similarity_threshold`) and update `get_method`:

```python
                "average_weights": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF: additive SUM, so per-LoRA strengths act as gains and stacked LoRAs keep full "
                               "magnitude (matches ComfyUI's native LoRA stacking and the other merge nodes, the "
                               "default). ON: normalized weighted AVERAGE, so strengths act as ratios and the result "
                               "is a blend/interpolation (weaker magnitude). Note: strengths control MAGNITUDE only, "
                               "not the interpolation position.",
                }),
```

```python
    def get_method(self, similarity_threshold: float = 0.001, average_weights: bool = False):
        method_def = {
            "name": "nearswap",
            "settings": {
                "similarity_threshold": similarity_threshold,
                "normalize": average_weights,
            }
        }
        return (method_def,)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_merge_node_names.py`
Expected: `All 6 passed`.

- [ ] **Step 5: Commit**

```bash
git add custom_nodes/LoRA-Merger-ComfyUI/src/nodes_merge_methods.py \
        custom_nodes/LoRA-Merger-ComfyUI/tests/test_merge_node_names.py
git commit -m "feat(nodes): average_weights toggle on slerp/nuslerp/karcher/nearswap"
```

---

## Task 3: Route the four modes through the delta-space branch

**Files:**
- Modify: `custom_nodes/LoRA-Merger-ComfyUI/src/lora_mergekit_merge.py`
- Test: `custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_integration.py` (new)

- [ ] **Step 1: Write the failing integration test**

Create `custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_integration.py`:

```python
# tests/test_interp_integration.py
# Standalone end-to-end test: run LoraMergerMergekit.merge() for the interpolation
# modes on CPU and confirm they produce a valid, non-zero merged LoRA.
import os, sys, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)
COMFY_ROOT = os.path.dirname(PARENT)
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, PARENT)

import torch
import importlib.util
PKG = "LoRA_Merger_ComfyUI_test"
spec = importlib.util.spec_from_file_location(
    PKG, os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
pkg = importlib.util.module_from_spec(spec)
sys.modules[PKG] = pkg
spec.loader.exec_module(pkg)

from LoRA_Merger_ComfyUI_test.src.lora_mergekit_merge import LoraMergerMergekit
from LoRA_Merger_ComfyUI_test.src.merge import get_merge_method, prepare_method_args


def _lora(seed, rank=8, out=64, inn=96):
    torch.manual_seed(seed)
    up = torch.randn(out, rank) * 0.1
    down = torch.randn(rank, inn) * 0.1
    return (up, down, torch.tensor(float(rank)))   # (up, down, alpha)


def _run(mode, settings, n_loras=2, key="lora_unet_test_layer"):
    node = LoraMergerMergekit()
    names = [f"loraA", f"loraB", f"loraC"][:n_loras]
    node.components = {key: {nm: _lora(i) for i, nm in enumerate(names)}}
    node.strengths = {nm: {"strength_model": 1.0, "strength_clip": 1.0} for nm in names}
    method = get_merge_method(mode)
    margs = prepare_method_args(mode, settings)
    result = node.merge(
        method=method, method_args=margs, lambda_=1.0, spectral_norm_scale=0.0,
        merge_clip=False, device=torch.device("cpu"), dtype=torch.float32)
    return result[0]   # merge() returns (lora_out,)


SETTINGS = {
    "slerp": {"t": 0.5, "normalize": False},
    "nuslerp": {"nuslerp_flatten": True, "nuslerp_row_wise": False, "normalize": False},
    "karcher": {"max_iter": 10, "tol": 1e-5, "normalize": False},
    "nearswap": {"similarity_threshold": 0.001, "normalize": False},
}


def test_each_mode_produces_nonzero_lora():
    for mode, st in SETTINGS.items():
        out = _run(mode, st)
        adapters = out["lora"]
        assert adapters, f"{mode}: empty adapter dict"
        for k, adapter in adapters.items():
            up, down, alpha = adapter.weights[0], adapter.weights[1], adapter.weights[2]
            assert torch.isfinite(up).all() and torch.isfinite(down).all(), f"{mode}: non-finite"
            recon = up @ down
            assert recon.norm().item() > 1e-6, f"{mode}: near-zero merge ({recon.norm()})"


def test_additive_stronger_than_average():
    # OFF (sum) should reconstruct a larger delta than ON (mean) for 2 LoRAs.
    def recon_norm(normalize):
        st = dict(SETTINGS["slerp"]); st["normalize"] = normalize
        out = _run("slerp", st)
        a = next(iter(out["lora"].values()))
        return (a.weights[0] @ a.weights[1]).norm().item()
    off = recon_norm(False)
    on = recon_norm(True)
    assert off > on * 1.5, f"additive({off}) not > average({on})"


def test_single_owner_key_not_zero():
    out = _run("slerp", SETTINGS["slerp"], n_loras=1)
    a = next(iter(out["lora"].values()))
    assert (a.weights[0] @ a.weights[1]).norm().item() > 1e-6


def run(tests):
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception:
            failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    if failed:
        print(f"\n{failed} FAILED"); sys.exit(1)
    print(f"\nAll {len(tests)} passed")


run([
    ("each_mode_nonzero", test_each_mode_produces_nonzero_lora),
    ("additive_stronger_than_average", test_additive_stronger_than_average),
    ("single_owner_not_zero", test_single_owner_key_not_zero),
])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_integration.py`
Expected: FAIL — the interpolation modes are not routed to delta-space yet, so they hit the factored path and (for `test_additive_stronger_than_average`) the `normalize`/magnitude behavior won't hold; `test_single_owner_key_not_zero` and reconstruction may also misbehave. (A failure of any of the three registered checks is the expected red state.)

- [ ] **Step 3: Update imports in `lora_mergekit_merge.py`**

In `custom_nodes/LoRA-Merger-ComfyUI/src/lora_mergekit_merge.py`, change the merge-package import block:

```python
from .merge import (
    create_map,
    create_tensor_param,
    get_merge_method,
    prepare_method_args,
    simple_weighted_average,
    sce_merge_deltas,
)
```

to add the two new names:

```python
from .merge import (
    create_map,
    create_tensor_param,
    get_merge_method,
    prepare_method_args,
    simple_weighted_average,
    sce_merge_deltas,
    interp_delta_merge,
    INTERP_MODES,
)
```

- [ ] **Step 4: Route the interpolation modes in the per-key branch**

In `lora_mergekit_merge.py`, find the delta-space routing (currently):

```python
            mode = method_args.get('mode')
            is_gta = (not is_clip) and (mode in GTA_MODES)
            # SCE also merges in delta space ...
            is_delta_space = is_gta or ((not is_clip) and mode == "sce")
```

Replace those lines with:

```python
            mode = method_args.get('mode')
            is_gta = (not is_clip) and (mode in GTA_MODES)
            # SCE and the interpolation methods (slerp/nuslerp/karcher/nearswap) also
            # merge in delta space: merging the up/down factors separately injects
            # meaningless up_i @ down_j cross-terms. Reconstruct each LoRA's full
            # delta, merge in delta space, then refactor back into a LoRA.
            is_interp = (not is_clip) and (mode in INTERP_MODES)
            is_delta_space = is_gta or is_interp or ((not is_clip) and mode == "sce")
```

Then find the block that computes `merged` inside `if is_delta_space:` (currently an `if is_gta: ... else:  # SCE in delta space ...`) and replace that if/else with a three-way branch:

```python
                if is_gta:
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
                elif is_interp:
                    # Interpolation methods: unit-weight blend + strength post-scale.
                    merged = interp_delta_merge(
                        method, deltas, weights.to(torch.float32), method_args,
                        key=key, normalize=method_args.get('normalize', False),
                        dtype=torch.float32,
                    )
                else:  # SCE in delta space
                    merged = sce_merge_deltas(
                        deltas,
                        weights.to(torch.float32),
                        select_topk=method_args.get('select_topk', 0.5),
                        int8_mask=method_args.get('int8_mask', False),
                        normalize=method_args.get('normalize', False),
                    )
```

(Only the `elif is_interp:` clause is new; keep the `if is_gta` and `else` SCE clauses exactly as they already are.)

- [ ] **Step 5: Add the interpolation modes to the CUDA worker-serialization guard**

In `lora_mergekit_merge.py`, find:

```python
        _mode = method_args.get('mode')
        is_delta_space = (_mode in GTA_MODES) or (_mode == "sce")
        n_workers = 1 if (is_delta_space and getattr(device, "type", device) == "cuda") else 8
```

Replace the predicate line with:

```python
        _mode = method_args.get('mode')
        is_delta_space = (_mode in GTA_MODES) or (_mode == "sce") or (_mode in INTERP_MODES)
        n_workers = 1 if (is_delta_space and getattr(device, "type", device) == "cuda") else 8
```

- [ ] **Step 6: Run the integration test to verify it passes**

Run: `.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_integration.py`
Expected: `All 3 passed`.

- [ ] **Step 7: Run the full standalone test suite (no regressions)**

Run each and confirm all pass:
```bash
.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_delta_merge.py
.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_merge_node_names.py
.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_gta_sparsify.py
```
Expected: `All 5 passed`, `All 6 passed`, `All 12 passed` respectively.

- [ ] **Step 8: Commit**

```bash
git add custom_nodes/LoRA-Merger-ComfyUI/src/lora_mergekit_merge.py \
        custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_integration.py
git commit -m "feat(merge): route slerp/nuslerp/karcher/nearswap through delta-space path"
```

---

## Task 4: Fidelity check (informational) + README note

**Files:**
- Test: `custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_fidelity.py` (new, informational)
- Modify: `custom_nodes/LoRA-Merger-ComfyUI/README.md`

- [ ] **Step 1: Add an informational fidelity script**

Create `custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_fidelity.py`:

```python
# tests/test_interp_fidelity.py
# Informational: delta-space blend should align better with the intended merge
# than the old factored path. Asserts a loose lower bound so it is not brittle.
import os, sys, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)
COMFY_ROOT = os.path.dirname(PARENT)
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, PARENT)

import torch
import importlib.util
PKG = "LoRA_Merger_ComfyUI_test"
spec = importlib.util.spec_from_file_location(
    PKG, os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
pkg = importlib.util.module_from_spec(spec)
sys.modules[PKG] = pkg
spec.loader.exec_module(pkg)

from LoRA_Merger_ComfyUI_test.src.merge.algorithms import (
    interp_delta_merge, slerp_merge, nuslerp_merge, karcher_merge)


def _delta(seed):
    torch.manual_seed(seed)
    return (torch.randn(96, 8) * 0.1) @ (torch.randn(8, 128) * 0.1)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def test_delta_space_blend_aligns_with_average():
    margs = {
        "slerp": (slerp_merge, {"mode": "slerp", "t": 0.5, "lambda_": 1.0}),
        "nuslerp": (nuslerp_merge, {"mode": "nuslerp", "nuslerp_flatten": True,
                                    "nuslerp_row_wise": False, "lambda_": 1.0}),
        "karcher": (karcher_merge, {"mode": "karcher", "max_iter": 10, "tol": 1e-5,
                                    "lambda_": 1.0}),
    }
    for name, (fn, ma) in margs.items():
        deltas = [_delta(1), _delta(2)]
        ref = 0.5 * (deltas[0] + deltas[1])            # the intended even blend
        out = interp_delta_merge(fn, [d.clone() for d in deltas], torch.tensor([1.0, 1.0]),
                                 dict(ma), key="k", normalize=True)   # average mode
        c = _cos(out, ref)
        print(f"  {name}: cos(delta-space blend, mean-delta) = {c:+.3f}")
        assert c > 0.6, f"{name}: unexpectedly low alignment {c}"


def run(tests):
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception:
            failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    if failed:
        print(f"\n{failed} FAILED"); sys.exit(1)
    print(f"\nAll {len(tests)} passed")


run([("delta_space_blend_aligns", test_delta_space_blend_aligns_with_average)])
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_fidelity.py`
Expected: `All 1 passed`, and printed cosines around 0.8–0.9.

- [ ] **Step 3: Update the README**

In `custom_nodes/LoRA-Merger-ComfyUI/README.md`, find the section describing the spherical/specialized merge methods (SLERP, NuSLERP, Karcher, NearSwap). Add a short note (place it near their descriptions):

```markdown
> **Delta-space + `average_weights` (v2.2.5):** SLERP, NuSLERP, Karcher and NearSwap now merge in
> delta space (like the GTA family), reconstructing the full LoRA delta instead of merging the
> up/down factors separately. Each also exposes `average_weights` (default **OFF** = additive, so
> stacked LoRAs keep full magnitude; **ON** = normalized average/blend). For these interpolation
> methods, per-LoRA strength controls **magnitude only** — blend position comes from the method's
> own parameter (SLERP `t`, NearSwap threshold).
```

- [ ] **Step 4: Commit**

```bash
git add custom_nodes/LoRA-Merger-ComfyUI/tests/test_interp_fidelity.py \
        custom_nodes/LoRA-Merger-ComfyUI/README.md
git commit -m "test(merge): interp delta-space fidelity check; docs(readme): note the change"
```

---

## Done criteria

- `average_weights` appears on SLERP/NuSLERP/Karcher/NearSwap nodes, default OFF, wired to `normalize`.
- Those four modes route through `interp_delta_merge` in the delta-space branch and are serialized on CUDA.
- All standalone tests pass: `test_interp_delta_merge.py` (5), `test_interp_integration.py` (3), `test_interp_fidelity.py` (1), `test_merge_node_names.py` (6), `test_gta_sparsify.py` (12).
- README documents the change.
```
