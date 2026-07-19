# tests/test_merge_node_names.py
# Standalone script test (repo pytest is broken). Loads the custom-node package
# under a synthetic name so the relative imports resolve, then asserts the
# renamed widget keys and the unchanged internal settings/context keys.
import importlib.util, os, sys, inspect, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)            # .../custom_nodes
COMFY_ROOT = os.path.dirname(PARENT)      # ComfyUI root, so `import comfy` resolves
PKG = "LoRA_Merger_ComfyUI_test"

sys.path.insert(0, COMFY_ROOT)            # lora_mergekit_merge imports comfy.*
sys.path.insert(0, PARENT)
spec = importlib.util.spec_from_file_location(
    PKG, os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
pkg = importlib.util.module_from_spec(spec)
sys.modules[PKG] = pkg
spec.loader.exec_module(pkg)

from LoRA_Merger_ComfyUI_test.src import nodes_merge_methods as N
from LoRA_Merger_ComfyUI_test.src.lora_mergekit_merge import LoraMergerMergekit

EXPECTED_ORDER = {
    "LinearMergeMethod": ["average_weights"],
    "TaskArithmeticMergeMethod": ["rescale_norm", "average_weights"],
    "TIESMergeMethod": ["rescale_norm", "average_weights", "density"],
    "DAREMergeMethod": ["sign_consensus", "rescale_norm", "density", "average_weights"],
    "BreadcrumbsMergeMethod": ["sign_consensus", "rescale_norm", "density", "gamma", "average_weights"],
    "DELLAMergeMethod": ["sign_consensus", "rescale_norm", "density", "epsilon", "average_weights"],
}


def test_input_types_renamed_same_order():
    for cls_name, expected in EXPECTED_ORDER.items():
        cls = getattr(N, cls_name)
        keys = list(cls.INPUT_TYPES()["required"].keys())
        assert keys == expected, f"{cls_name}: {keys} != {expected}"


def test_settings_internal_keys_unchanged():
    # average_weights -> internal "normalize"; sign_consensus -> "sign_consensus_algorithm"
    for cls_name in EXPECTED_ORDER:
        cls = getattr(N, cls_name)
        s = cls().get_method(average_weights=False)[0]["settings"]
        assert s["normalize"] is False, f"{cls_name}: normalize not wired from average_weights"
    for cls_name in ("DAREMergeMethod", "BreadcrumbsMergeMethod", "DELLAMergeMethod"):
        cls = getattr(N, cls_name)
        s = cls().get_method(sign_consensus=True)[0]["settings"]
        assert s["sign_consensus_algorithm"] is True, f"{cls_name}: sign_consensus not wired"


def test_merger_output_scale():
    keys = list(LoraMergerMergekit.INPUT_TYPES()["required"].keys())
    expected = ["method", "components", "strengths", "output_scale",
                "spectral_norm_scale", "merge_clip", "device", "dtype", "refactor_method"]
    assert keys == expected, f"merger widget order {keys} != {expected}"
    params = inspect.signature(LoraMergerMergekit.lora_mergekit).parameters
    assert "output_scale" in params and "lambda_" not in params, list(params)


def run():
    failed = 0
    for name, fn in [
        ("input_types_renamed_same_order", test_input_types_renamed_same_order),
        ("settings_internal_keys_unchanged", test_settings_internal_keys_unchanged),
        ("merger_output_scale", test_merger_output_scale),
    ]:
        try:
            fn(); print(f"PASS {name}")
        except Exception:
            failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    if failed:
        print(f"\n{failed} FAILED"); sys.exit(1)
    print("\nAll 3 passed")


run()
