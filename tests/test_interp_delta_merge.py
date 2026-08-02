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


# Runnable as a plain script (`python tests/<file>.py`); under pytest the
# test_* functions are collected directly, so the script runner must not fire
# at import time -- it calls sys.exit() and would abort collection.
if __name__ == "__main__":
    run([
        ("modes_constant", test_modes_constant),
        ("additive_is_blend_times_sum", test_additive_is_blend_times_sum),
        ("average_is_blend_times_mean", test_average_is_blend_times_mean),
        ("strength_linear_gain_additive", test_strength_is_linear_gain_additive),
        ("single_owner_fallback", test_single_owner_fallback_not_zero),
    ])
