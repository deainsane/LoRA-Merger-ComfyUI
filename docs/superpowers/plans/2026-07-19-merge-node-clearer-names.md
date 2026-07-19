# Clearer Merge-Node Widget Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename three cryptic user-facing widget keys (`ties → sign_consensus`, `normalize → average_weights`, `lambda_ → output_scale`) with no behavior change.

**Architecture:** Rename only the widget key + the node's entry parameter; keep every internal settings/context key (`sign_consensus_algorithm`, `normalize`, `lambda_`) so the merge pipeline is untouched. ComfyUI stores `widgets_values` positionally, so same-position renames preserve saved graph workflows.

**Tech Stack:** Python, ComfyUI custom-node API. Repo pytest is broken; tests run as standalone scripts via `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/<file>.py`.

---

## Reference: exact current widget order (must be preserved except for the renamed key)

| node | current required-widget order |
| --- | --- |
| LinearMergeMethod | `normalize` |
| TaskArithmeticMergeMethod | `rescale_norm, normalize` |
| TIESMergeMethod | `rescale_norm, normalize, density` |
| DAREMergeMethod | `ties, rescale_norm, density, normalize` |
| BreadcrumbsMergeMethod | `ties, rescale_norm, density, gamma, normalize` |
| DELLAMergeMethod | `ties, rescale_norm, density, epsilon, normalize` |
| LoraMergerMergekit | `method, components, strengths, lambda_, spectral_norm_scale, merge_clip, device, dtype, refactor_method` |

Structural note: `TaskArithmetic/TIES/DARE/Breadcrumbs/DELLA` use an **inline `get_method`**; `Linear` uses the base-class `get_method` that calls its own **`get_settings`**. The `ties` widget exists only on DARE, Breadcrumbs, DELLA.

---

### Task 1: Guarding test (write first, must fail)

**Files:**
- Create: `tests/test_merge_node_names.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge_node_names.py
# Standalone script test (repo pytest is broken). Loads the custom-node package
# under a synthetic name so the relative imports resolve, then asserts the
# renamed widget keys and the unchanged internal settings/context keys.
import importlib.util, os, sys, inspect, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)
PKG = "LoRA_Merger_ComfyUI_test"

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_merge_node_names.py 2>&1 | grep -v FutureWarning | grep -v pynvml | grep -v "Loading:"`
Expected: FAIL — current keys are `normalize`/`ties`/`lambda_`, so `test_input_types_renamed_same_order` and `test_merger_output_scale` fail.

- [ ] **Step 3: Commit the test**

```bash
git add tests/test_merge_node_names.py
git commit -m "test(nodes): guard clearer widget names (ties/normalize/lambda_ renames)"
```

---

### Task 2: Rename the boolean widgets in `nodes_merge_methods.py`

**Files:**
- Modify: `src/nodes_merge_methods.py`

All edits below are exact, unambiguous find/replace pairs. Apply each with the Edit tool using `replace_all` where a count is given.

- [ ] **Step 1: Rename the widget KEYS in INPUT_TYPES**

Replace (replace_all, 3 occurrences):
`"ties": ("BOOLEAN"` → `"sign_consensus": ("BOOLEAN"`

Replace (replace_all, 6 occurrences):
`"normalize": ("BOOLEAN"` → `"average_weights": ("BOOLEAN"`

- [ ] **Step 2: Rename the parameters in the signatures**

Replace (replace_all, 3 occurrences):
`self, ties: bool = False, rescale_norm` → `self, sign_consensus: bool = False, rescale_norm`

Replace (replace_all, 6 occurrences):
`normalize: bool` → `average_weights: bool`

- [ ] **Step 3: Rewire the emitted settings values (keys stay the same)**

Replace (replace_all, 3 occurrences):
`"sign_consensus_algorithm": ties,` → `"sign_consensus_algorithm": sign_consensus,`

Replace (replace_all, 5 occurrences):
`"normalize": normalize,` → `"normalize": average_weights,`

Replace (single occurrence, Linear's `get_settings`):
`return {"normalize": normalize}` → `return {"normalize": average_weights}`

- [ ] **Step 4: Run the test — boolean assertions pass, merger still fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_merge_node_names.py 2>&1 | grep -v FutureWarning | grep -v pynvml | grep -v "Loading:"`
Expected: `PASS input_types_renamed_same_order`, `PASS settings_internal_keys_unchanged`, `FAIL merger_output_scale`.

- [ ] **Step 5: Commit**

```bash
git add src/nodes_merge_methods.py
git commit -m "ux(nodes): rename ties->sign_consensus, normalize->average_weights widgets"
```

---

### Task 3: Rename `lambda_ → output_scale` on the Merger node

**Files:**
- Modify: `src/lora_mergekit_merge.py`

Only the widget key and the entry parameter are renamed; the internal keyword `lambda_=` at the `self.merge(...)` call and the `"lambda_"` merge-context key are kept (their *values* switch to `output_scale`).

- [ ] **Step 1: Rename the widget key + clarify its tooltip**

Replace:
```python
                "lambda_": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Lambda value for scaling the merged model.",
                }),
```
with:
```python
                "output_scale": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Global scale (0-1) applied to the final merged result.",
                }),
```

- [ ] **Step 2: Rename the entry parameter**

Replace:
`                      lambda_: float = 1.0,` → `                      output_scale: float = 1.0,`

- [ ] **Step 3: Feed it to the internal pipeline under the unchanged key**

Replace:
`        merge = self.merge(method=merge_method, method_args=method_args, lambda_=lambda_,`
with:
`        merge = self.merge(method=merge_method, method_args=method_args, lambda_=output_scale,`

Replace:
`            "lambda_": lambda_,` → `            "lambda_": output_scale,`

- [ ] **Step 4: Run the test — all pass**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_merge_node_names.py 2>&1 | grep -v FutureWarning | grep -v pynvml | grep -v "Loading:"`
Expected: `All 3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/lora_mergekit_merge.py
git commit -m "ux(nodes): rename lambda_->output_scale widget on the Merger node"
```

---

### Task 4: Full-suite regression + package import sanity

**Files:** none (verification only)

- [ ] **Step 1: Re-run the GTA behavior/sparsify suites (internal keys unchanged → must still pass)**

Run:
```bash
cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI
for t in tests/test_gta_sparsify.py tests/test_gta_behavior.py tests/test_merge_node_names.py; do
  echo "== $t =="; /home/lars/SD/Apps/ComfyUI/.venv/bin/python "$t" 2>&1 | grep -v FutureWarning | grep -v pynvml | grep -v "Loading:" | tail -3
done
```
Expected: every suite prints `All N passed`.

- [ ] **Step 2: Confirm the package still imports (node registration intact)**

Run:
```bash
cd /home/lars/SD/Apps/ComfyUI && /home/lars/SD/Apps/ComfyUI/.venv/bin/python -c "
import importlib.util, sys, os
root='custom_nodes/LoRA-Merger-ComfyUI'; sys.path.insert(0, os.path.dirname(os.path.abspath(root)))
s=importlib.util.spec_from_file_location('pm', os.path.join(root,'__init__.py'), submodule_search_locations=[root])
m=importlib.util.module_from_spec(s); sys.modules['pm']=m; s.loader.exec_module(m)
print('nodes registered:', len(m.NODE_CLASS_MAPPINGS))
" 2>&1 | grep -v FutureWarning | grep -v pynvml
```
Expected: `nodes registered: 26` (or the current count) with no traceback.

- [ ] **Step 3: No commit needed** — verification only. The work is complete when all suites pass and the package imports.

---

## Notes / out of scope

- `src/experimental/checkpoint_merge.py` has its own separate `lambda_` widget on a different node — **not** touched here.
- `src/merge/base_node.py` `BaseTaskArithmeticNode` (shared `get_settings` with `normalize`) is **not** used by the active method nodes (they define inline `get_method`); leave it unchanged to avoid scope creep.
- The buggy `normalize: bool = 0.5` default in `TaskArithmeticMergeMethod.get_method` becomes `average_weights: bool = 0.5` — the widget default (`True`) governs the UI, so this latent oddity is carried as-is (naming-only scope).