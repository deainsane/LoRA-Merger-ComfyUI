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
        ref = 0.5 * (deltas[0] + deltas[1])
        out = interp_delta_merge(fn, [d.clone() for d in deltas], torch.tensor([1.0, 1.0]),
                                 dict(ma), key="k", normalize=True)
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


# Runnable as a plain script (`python tests/<file>.py`); under pytest the
# test_* functions are collected directly, so the script runner must not fire
# at import time -- it calls sys.exit() and would abort collection.
if __name__ == "__main__":
    run([("delta_space_blend_aligns", test_delta_space_blend_aligns_with_average)])
