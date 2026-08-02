# tests/test_lora_save.py
# Standalone script test (repo pytest collection is broken). Loads the custom-node
# package under a synthetic name so the relative imports resolve, then checks that
# save-time sanitation turns the transposed / sliced views produced by the refactor
# paths into tensors safetensors will actually serialize.
import importlib.util, os, sys, tempfile, traceback

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)            # .../custom_nodes
COMFY_ROOT = os.path.dirname(PARENT)      # ComfyUI root, so `import comfy` resolves
PKG = "LoRA_Merger_ComfyUI_test"

sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, PARENT)
spec = importlib.util.spec_from_file_location(
    PKG, os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
pkg = importlib.util.module_from_spec(spec)
sys.modules[PKG] = pkg
spec.loader.exec_module(pkg)

from LoRA_Merger_ComfyUI_test.src.lora_save import sanitize_for_save


def storage_elems(t):
    return t.untyped_storage().nbytes() // t.element_size()


def test_transposed_view_becomes_contiguous():
    # e.g. down = (V[:, :r] * s_sqrt).T
    view = torch.randn(8, 4).T
    assert not view.is_contiguous()

    out = sanitize_for_save(view)

    assert out.is_contiguous()
    assert out.shape == view.shape
    assert torch.equal(out, view)


def test_column_slice_drops_shared_storage():
    # e.g. up = U[:, :r] out of a q-column randomized SVD result
    base = torch.randn(16, 8)
    view = base[:, :3]

    out = sanitize_for_save(view)

    assert out.is_contiguous()
    assert torch.equal(out, view)
    assert storage_elems(out) == out.numel()


def test_row_slice_drops_shared_storage():
    # Contiguous view into a larger storage: .contiguous() alone is a no-op here,
    # and safetensors rejects it as a shared-storage tensor.
    base = torch.randn(16, 8)
    view = base[:4]
    assert view.is_contiguous()

    out = sanitize_for_save(view)

    assert torch.equal(out, view)
    assert storage_elems(out) == out.numel()
    assert out.data_ptr() != base.data_ptr()


def test_dense_tensor_is_passed_through_without_copy():
    t = torch.randn(4, 4)
    assert sanitize_for_save(t).data_ptr() == t.data_ptr()


def test_non_tensor_values_pass_through():
    assert sanitize_for_save(1.0) == 1.0
    assert sanitize_for_save(None) is None


def test_sanitized_state_dict_is_serializable():
    import safetensors.torch as st

    base = torch.randn(16, 8)
    state_dict = {
        "lora_unet_blocks_4_mlp_down.lora_up.weight": base[:, :3],
        "lora_unet_blocks_4_mlp_down.lora_down.weight": torch.randn(8, 3).T,
        "lora_unet_blocks_4_mlp_down.alpha": torch.tensor(3.0),
    }

    with tempfile.TemporaryDirectory() as tmp:
        # Regression guard: the unsanitized dict is exactly what used to blow up.
        raised = False
        try:
            st.save_file(state_dict, os.path.join(tmp, "raw.safetensors"))
        except ValueError as e:
            raised = "non contiguous" in str(e) or "shared" in str(e).lower()
        assert raised, "expected safetensors to reject the raw views"

        path = os.path.join(tmp, "sanitized.safetensors")
        st.save_file({k: sanitize_for_save(v) for k, v in state_dict.items()}, path)

        loaded = st.load_file(path)
        assert set(loaded) == set(state_dict)
        for k, v in state_dict.items():
            assert torch.equal(loaded[k], v)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name[5:]}")
        except Exception:
            failures += 1
            print(f"FAIL {name[5:]}")
            traceback.print_exc()
    print(f"\n{'All' if not failures else failures} " + ("passed" if not failures else "failed"))
    sys.exit(1 if failures else 0)
