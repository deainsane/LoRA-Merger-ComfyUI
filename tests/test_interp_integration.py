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
    return (up, down, torch.tensor(float(rank)))


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
    return result[0]


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


# Runnable as a plain script (`python tests/<file>.py`); under pytest the
# test_* functions are collected directly, so the script runner must not fire
# at import time -- it calls sys.exit() and would abort collection.
if __name__ == "__main__":
    run([
        ("each_mode_nonzero", test_each_mode_produces_nonzero_lora),
        ("additive_stronger_than_average", test_additive_stronger_than_average),
        ("single_owner_not_zero", test_single_owner_key_not_zero),
    ])
