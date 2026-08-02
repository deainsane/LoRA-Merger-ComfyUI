# tests/test_merger_vram_offload.py
# Verifies the PM LoRA Merger's `offload_models` widget: it evicts resident
# models from VRAM (comfy.model_management.unload_all_models) before the merge
# only when enabled AND the merge device is cuda. Uses the same standalone
# package-loader as test_merge_node_names so the node's relative imports resolve.
import importlib.util, os, sys, inspect
from unittest.mock import patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)            # .../custom_nodes
COMFY_ROOT = os.path.dirname(PARENT)      # ComfyUI root, so `import comfy` resolves
PKG = "LoRA_Merger_ComfyUI_offload_test"

sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, PARENT)
spec = importlib.util.spec_from_file_location(
    PKG, os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
pkg = importlib.util.module_from_spec(spec)
sys.modules[PKG] = pkg
spec.loader.exec_module(pkg)

from LoRA_Merger_ComfyUI_offload_test.src import lora_mergekit_merge as mod
LoraMergerMergekit = mod.LoraMergerMergekit


class _StopAfterOffload(Exception):
    """Sentinel raised in place of the merge so we test only the offload step."""


def _run(offload_models, device):
    """Call lora_mergekit far enough to execute the offload guard, then bail.

    Returns the unload_all_models mock so the caller can assert on it. We patch
    `mod.comfy.model_management` (the exact object the node calls through) so the
    test is valid whether comfy is the real module or a conftest MagicMock, and
    never touches real CUDA/model state.
    """
    mm = mod.comfy.model_management
    with patch.object(mod, "get_merge_method", side_effect=_StopAfterOffload), \
         patch.object(mm, "unload_all_models") as unload, \
         patch.object(mm, "soft_empty_cache"):
        try:
            LoraMergerMergekit().lora_mergekit(
                method={"name": "linear", "settings": {}},
                components={"dummy.layer": {}},
                strengths={},
                device=device,
                dtype="float32",
                offload_models=offload_models,
            )
        except _StopAfterOffload:
            pass
    return unload


def test_offload_widget_present_default_true():
    req = LoraMergerMergekit.INPUT_TYPES()["required"]
    assert "offload_models" in req, "offload_models widget missing"
    spec_tuple = req["offload_models"]
    assert spec_tuple[0] == "BOOLEAN", f"expected BOOLEAN, got {spec_tuple[0]!r}"
    assert spec_tuple[1]["default"] is True, "offload_models should default to True"


def test_offload_param_default_true():
    params = inspect.signature(LoraMergerMergekit.lora_mergekit).parameters
    assert "offload_models" in params, "offload_models param missing from signature"
    assert params["offload_models"].default is True, "offload_models default should be True"


def test_unloads_on_cuda_when_enabled():
    assert _run(offload_models=True, device="cuda").called, \
        "unload_all_models should be called for cuda merge with offload_models=True"


def test_no_unload_when_disabled():
    assert not _run(offload_models=False, device="cuda").called, \
        "unload_all_models must not be called when offload_models=False"


def test_no_unload_on_cpu():
    assert not _run(offload_models=True, device="cpu").called, \
        "unload_all_models must not be called for a cpu merge"


def run():
    import traceback
    tests = [
        ("offload_widget_present_default_true", test_offload_widget_present_default_true),
        ("offload_param_default_true", test_offload_param_default_true),
        ("unloads_on_cuda_when_enabled", test_unloads_on_cuda_when_enabled),
        ("no_unload_when_disabled", test_no_unload_when_disabled),
        ("no_unload_on_cpu", test_no_unload_on_cpu),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception:
            failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    run()
