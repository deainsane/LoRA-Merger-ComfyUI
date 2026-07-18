# Block Selector lora_stack Removal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `BlockSelector` self-contained by removing the `lora_stack` input. BlockSelector outputs `(index, BlockDefinition)` tuples that `LoraDecompose` resolves using its own `key_dicts`.

**Architecture:** Two new helpers in `blocks.py` (`build_block_selection_dict`, `resolve_block_selection`) separate the concerns cleanly: BlockSelector builds a `{configs: {index: definition}}` dict; LoraDecompose resolves indices to lora names and computes per-key weights.

**Tech Stack:** Pure Python (no ComfyUI imports in `blocks.py`).

---

## File Map

| File | Change |
|------|--------|
| `src/blocks.py` | Add `build_block_selection_dict`, `resolve_block_selection`. Keep all existing functions. |
| `src/nodes_block_selector.py` | Refactor `BlockSelector` to use `build_block_selection_dict`. Remove `lora_stack` input. |
| `src/lora_decompose.py` | Call `resolve_block_selection` before the per-key loop. |
| `src/types.py` | Add `BlockSelectionConfig` type alias. |
| `tests/test_blocks.py` | Rewrite `TestApplySelection` tests. Add `TestBuildBlockSelectionDict` and `TestResolveBlockSelection` tests. |

---

## Task 1: Add helpers to `src/blocks.py`

**Files:**
- Modify: `src/blocks.py` (add after `merge_selection`, before `apply_selection`)
- Test: `tests/test_blocks.py`

- [ ] **Step 1: Write failing tests for new helpers**

Open `tests/test_blocks.py` and add after the `TestMergeSelection` class (before `TestApplySelection`):

```python
from blocks import build_block_selection_dict, resolve_block_selection

class TestBuildBlockSelectionDict:
    def test_first_node_returns_configs_with_none_chain(self):
        result = build_block_selection_dict(None, index=0, definition=KREA2_DEF)
        assert result == {"configs": {0: KREA2_DEF}, "chain": None}

    def test_chaining_adds_second_index(self):
        chain = {"configs": {0: KREA2_DEF}, "chain": None}
        result = build_block_selection_dict(chain, index=1, definition=KREA2_DEF)
        assert result == {"configs": {0: KREA2_DEF, 1: KREA2_DEF}, "chain": chain}

    def test_chain_is_preserved(self):
        chain = {"configs": {0: KREA2_DEF}, "chain": {"configs": {2: KREA2_DEF}, "chain": None}}
        result = build_block_selection_dict(chain, index=1, definition=KREA2_DEF)
        assert result["chain"] is chain

    def test_index_collision_raises(self):
        chain = {"configs": {0: KREA2_DEF}, "chain": None}
        import pytest
        with pytest.raises(ValueError, match="already has a config"):
            build_block_selection_dict(chain, index=0, definition=KREA2_DEF)

    def test_negative_index_raises(self):
        import pytest
        with pytest.raises(ValueError, match="negative"):
            build_block_selection_dict(None, index=-1, definition=KREA2_DEF)


class TestResolveBlockSelection:
    def _keys_by_name(self):
        return OrderedDict([
            ("lora0", ["diffusion_model.blocks.25.attn.wq.weight"]),
            ("lora1", ["diffusion_model.blocks.3.attn.wq.weight"]),
        ])

    def test_resolves_index_to_lora_name(self):
        selection = {"configs": {0: KREA2_DEF}, "chain": None}
        lora_names = list(self._keys_by_name().keys())
        out = resolve_block_selection(selection, lora_names)
        assert out == {"lora0": {"diffusion_model.blocks.25.attn.wq.weight": 0.0}}

    def test_resolves_different_indices(self):
        selection = {"configs": {0: KREA2_DEF, 1: KREA2_DEF}, "chain": None}
        lora_names = list(self._keys_by_name().keys())
        out = resolve_block_selection(selection, lora_names)
        assert set(out.keys()) == {"lora0", "lora1"}

    def test_out_of_range_index_skipped_with_warning(self, caplog):
        selection = {"configs": {9: KREA2_DEF}, "chain": None}
        lora_names = ["lora0", "lora1"]
        out = resolve_block_selection(selection, lora_names)
        assert out is None
        assert "out of range" in caplog.text

    def test_empty_configs_returns_none(self):
        out = resolve_block_selection({"configs": {}, "chain": None}, ["lora0"])
        assert out is None

    def test_none_selection_returns_none(self):
        out = resolve_block_selection(None, ["lora0"])
        assert out is None

    def test_empty_lora_names_returns_none(self):
        out = resolve_block_selection({"configs": {0: KREA2_DEF}, "chain": None}, [])
        assert out is None
```

- [ ] **Step 2: Run tests to verify they fail (missing functions)**

Run: `cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI && python -m pytest tests/test_blocks.py::TestBuildBlockSelectionDict -v 2>&1 | head -20`
Expected: FAIL — `build_block_selection_dict` not defined

Run: `cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI && python -m pytest tests/test_blocks.py::TestResolveBlockSelection -v 2>&1 | head -20`
Expected: FAIL — `resolve_block_selection` not defined

- [ ] **Step 3: Add `build_block_selection_dict` to `src/blocks.py`**

Insert after `merge_selection` (after line 89) and before `apply_selection` (line 92):

```python
def build_block_selection_dict(
    chain: Optional[dict],
    index: int,
    definition: Dict[str, Any]
) -> dict:
    """Build a BlockSelection dict by merging a new (index, definition) pair into chain.

    ``chain`` is the previous BlockSelection dict (or None). Raises ValueError on
    index collision or negative index.
    """
    if index < 0:
        raise ValueError(f"[PM Block Selector] index must be non-negative, got {index}")
    new_configs = {index: definition}
    if chain is None:
        return {"configs": new_configs, "chain": None}
    if index in chain["configs"]:
        raise ValueError(
            f"[PM Block Selector] index {index} already has a config "
            f"(chaining two BlockSelectors with the same index is not allowed)"
        )
    merged = dict(chain["configs"])
    merged.update(new_configs)
    return {"configs": merged, "chain": chain}
```

- [ ] **Step 4: Add `resolve_block_selection` to `src/blocks.py`**

Insert after `build_block_selection_dict` and before `apply_selection`:

```python
def resolve_block_selection(
    selection: Optional[dict],
    lora_names: List[str]
) -> Optional[Dict[str, Dict[str, float]]]:
    """Resolve a BlockSelection dict (index -> BlockDefinition) to the format
    expected by ``apply_block_weights``: {lora_name: {key: weight}}.

    Returns None if ``selection`` is None, empty, or resolves to nothing.
    Logs a warning for out-of-range indices (those configs are skipped).
    """
    if not selection or not lora_names:
        return None
    configs = selection.get("configs", {})
    if not configs:
        return None
    result = {}
    for idx, definition in configs.items():
        if idx < 0 or idx >= len(lora_names):
            logging.warning(
                f"[PM Block Selector] index {idx} out of range (0..{len(lora_names) - 1}); skipping."
            )
            continue
        lora_name = lora_names[idx]
        weights = compute_lora_weights(
            list(analyse_keys_for_lora(selection, lora_name)), definition
        )
        if weights:
            result[lora_name] = weights
    return result if result else None


def analyse_keys_for_lora(selection: dict, lora_name: str) -> Iterable[Any]:
    """Placeholder — resolve lora keys from selection. Currently returns [] because
    the new BlockSelector no longer carries lora keys; LoraDecompose passes real keys."""
    return []
```

> **Note:** `analyse_keys_for_lora` is a temporary stub. `LoraDecompose` will pass the actual keys via a different mechanism (see Task 4). For now, the test `test_resolves_index_to_lora_name` needs the actual keys. Update `resolve_block_selection` to accept `keys_by_name: Dict[str, List[Any]]` instead:

Update the function signature and body:

```python
def resolve_block_selection(
    selection: Optional[dict],
    keys_by_name: Dict[str, List[Any]]
) -> Optional[Dict[str, Dict[str, float]]]:
    """Resolve a BlockSelection dict (index -> BlockDefinition) to the format
    expected by ``apply_block_weights``: {lora_name: {key: weight}}.

    ``keys_by_name`` maps lora_name to its layer keys (from LoraDecompose.key_dicts).
    Returns None if ``selection`` is None, empty, or resolves to nothing.
    Logs a warning for out-of-range indices (those configs are skipped).
    """
    if not selection or not keys_by_name:
        return None
    lora_names = list(keys_by_name.keys())
    configs = selection.get("configs", {})
    if not configs:
        return None
    result = {}
    for idx, definition in configs.items():
        if idx < 0 or idx >= len(lora_names):
            logging.warning(
                f"[PM Block Selector] index {idx} out of range (0..{len(lora_names) - 1}); skipping."
            )
            continue
        lora_name = lora_names[idx]
        weights = compute_lora_weights(keys_by_name[lora_name], definition)
        if weights:
            result[lora_name] = weights
    return result if result else None
```

And update the tests accordingly (pass `keys_by_name` dict instead of `lora_names` list):

```python
class TestResolveBlockSelection:
    def _keys_by_name(self):
        return OrderedDict([
            ("lora0", ["diffusion_model.blocks.25.attn.wq.weight"]),
            ("lora1", ["diffusion_model.blocks.3.attn.wq.weight"]),
        ])

    def test_resolves_index_to_lora_name(self):
        selection = {"configs": {0: KREA2_DEF}, "chain": None}
        out = resolve_block_selection(selection, self._keys_by_name())
        assert out == {"lora0": {"diffusion_model.blocks.25.attn.wq.weight": 0.0}}

    def test_resolves_different_indices(self):
        selection = {"configs": {0: KREA2_DEF, 1: KREA2_DEF}, "chain": None}
        out = resolve_block_selection(selection, self._keys_by_name())
        assert set(out.keys()) == {"lora0", "lora1"}

    def test_out_of_range_index_skipped_with_warning(self, caplog):
        selection = {"configs": {9: KREA2_DEF}, "chain": None}
        out = resolve_block_selection(selection, dict(self._keys_by_name()))
        assert out is None
        assert "out of range" in caplog.text

    def test_empty_configs_returns_none(self):
        out = resolve_block_selection({"configs": {}, "chain": None}, self._keys_by_name())
        assert out is None

    def test_none_selection_returns_none(self):
        out = resolve_block_selection(None, self._keys_by_name())
        assert out is None

    def test_empty_keys_by_name_returns_none(self):
        out = resolve_block_selection({"configs": {0: KREA2_DEF}, "chain": None}, OrderedDict())
        assert out is None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI && python -m pytest tests/test_blocks.py::TestBuildBlockSelectionDict tests/test_blocks.py::TestResolveBlockSelection -v`
Expected: PASS for all 12 tests.

- [ ] **Step 6: Commit**

```bash
git add src/blocks.py tests/test_blocks.py
git commit -m "feat(blocks): add build_block_selection_dict and resolve_block_selection"
```

---

## Task 2: Refactor `src/nodes_block_selector.py`

**Files:**
- Modify: `src/nodes_block_selector.py` (lines 63-92)
- Test: no new tests needed (logic is tested via Task 1 helpers)

- [ ] **Step 1: Write the refactored `select` method**

Replace the entire `select` method body (lines 87-92) with:

```python
def select(self, block_definition, index, block_selection=None):
    if index < 0:
        raise ValueError("[PM Block Selector] index must be non-negative")
    result = build_block_selection_dict(block_selection, index, block_definition)
    logging.info(f"[PM Block Selector] index {index}: selection now covers "
                 f"{len(result['configs'])} LoRA(s).")
    return (result,)
```

- [ ] **Step 2: Update `INPUT_TYPES` — remove `lora_stack`**

In `INPUT_TYPES()` (lines 74-85), remove `"lora_stack": ("LoRAStack",),` from the required dict:

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "block_definition": ("BlockDefinition",),
            "index": ("INT", {"default": 0, "min": 0, "max": 1000,
                              "tooltip": "Which LoRA in the stack (0-based) to weight."}),
        },
        "optional": {
            "block_selection": ("BlockSelection",),
        },
    }
```

- [ ] **Step 3: Update `DESCRIPTION` to reflect new behavior**

Replace the `DESCRIPTION` string with:

```python
DESCRIPTION = ("Outputs a block selection config for the LoRA at 'index' in the LoRAStack. "
               "Chain block_selection outputs to weight multiple LoRAs. "
               "Feed the result into PM LoRA Stack Decompose.")
```

- [ ] **Step 4: Run existing tests to ensure nothing broke**

Run: `cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI && python -m pytest tests/test_blocks.py -v`
Expected: All tests pass (existing tests cover pure logic unaffected by the node refactor).

- [ ] **Step 5: Commit**

```bash
git add src/nodes_block_selector.py
git commit -m "refactor(BlockSelector): remove lora_stack input, chain index-based configs"
```

---

## Task 3: Update `src/types.py`

**Files:**
- Modify: `src/types.py` (add to imports and exports)

- [ ] **Step 1: Add `BlockSelectionConfig` type alias**

In the `Core LoRA Tensor Types` section (around line 25), add after the existing type aliases:

```python
# BlockSelection dict: maps stack index -> BlockDefinition (from blocks.py)
BlockSelectionConfig = Dict[int, Dict[str, Any]]
```

- [ ] **Step 2: Add to `__all__`**

Add `"BlockSelectionConfig"` to the `__all__` list (around line 317).

- [ ] **Step 3: Commit**

```bash
git add src/types.py
git commit -m "types: add BlockSelectionConfig alias"
```

---

## Task 4: Update `src/lora_decompose.py`

**Files:**
- Modify: `src/lora_decompose.py` (lines 11, 184-188)
- Test: no new tests needed (logic is tested via Task 1)

- [ ] **Step 1: Update import**

Line 11: change `from .blocks import apply_block_weights, apply_selection, normalize_key` to:

```python
from .blocks import apply_block_weights, build_block_selection_dict, normalize_key, resolve_block_selection
```

> Actually, `build_block_selection_dict` is not needed in lora_decompose — only `resolve_block_selection`. Fix:

```python
from .blocks import apply_block_weights, normalize_key, resolve_block_selection
```

- [ ] **Step 2: Call `resolve_block_selection` before the per-key loop**

In the `decompose` method, after line 179 (`logging.info(...)`) and before line 181 (`pbar = comfy.utils.ProgressBar(...)`), add:

```python
# Resolve index-based BlockSelection to lora-name-based {lora_name: {key: weight}}
keys_by_name = {name: list(lora_key_dict.keys()) for name, lora_key_dict in key_dicts.items()}
resolved_selection = resolve_block_selection(block_selection, keys_by_name)
```

Then change the call inside `process_key` (line 188) from:
```python
uda = apply_block_weights(uda, normalize_key(key), block_selection)
```
to:
```python
uda = apply_block_weights(uda, normalize_key(key), resolved_selection)
```

Also update the import at line 11 — `apply_selection` is no longer used there, but it's fine to leave it since it was imported but may or may not still be used. Check if `apply_selection` is used in `lora_decompose.py` — it's only used in the import and was passed to `apply_block_weights` in the old code, which is now changed to `resolved_selection`. So `apply_selection` can be removed from the import.

- [ ] **Step 3: Commit**

```bash
git add src/lora_decompose.py
git commit -m "refactor(LoraDecompose): use resolve_block_selection for index-based block selection"
```

---

## Task 5: Clean up test file

**Files:**
- Modify: `tests/test_blocks.py`

- [ ] **Step 1: Remove old `TestApplySelection` tests**

Delete the entire `TestApplySelection` class (lines 125-148). These tested the old `apply_selection` function which is no longer used by the node.

> Keep `TestMergeSelection` — `merge_selection` is still a valid helper and still tested. Keep all other test classes unchanged.

- [ ] **Step 2: Add `TestBuildBlockSelectionDict` and `TestResolveBlockSelection`**

These were already added in Task 1, Step 1. Verify they are present and pass.

- [ ] **Step 3: Run full test suite**

Run: `cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI && python -m pytest tests/test_blocks.py -v`
Expected: All tests pass (excluding any deleted test names).

- [ ] **Step 4: Commit**

```bash
git add tests/test_blocks.py
git commit -m "tests(blocks): rewrite TestApplySelection, add BlockSelection dict/resolve tests"
```

---

## Spec Coverage Check

| Spec requirement | Task |
|------------------|------|
| BlockSelector removes lora_stack input | Task 2 |
| BlockSelector validates negative index | Task 2 |
| BlockSelector raises on index collision | Task 1 (`build_block_selection_dict`) |
| BlockSelector chains via `block_selection` input | Task 2 |
| BlockSelector outputs `{configs: {index: definition}, chain: prev}` | Task 1 (`build_block_selection_dict`) |
| `resolve_block_selection` converts index → lora name | Task 1 (`resolve_block_selection`) |
| `resolve_block_selection` handles out-of-range with warning | Task 1 tests |
| `resolve_block_selection` returns None for empty/no selection | Task 1 tests |
| LoraDecompose calls resolve before per-key loop | Task 4 |
| Type definition added | Task 3 |
| Tests for new helpers | Task 1 + Task 5 |

All spec items covered. No gaps.

---

## Placeholder Scan

No TBD/TODO found. All code steps show actual implementation.

## Type Consistency

- `build_block_selection_dict(chain, index, definition)` — consistent across Tasks 1 and 2
- `resolve_block_selection(selection, keys_by_name)` — Dict[str, List[Any]] passed from Task 4
- `keys_by_name` is built as `{name: list(lora_key_dict.keys())` in Task 4, matching the Dict[str, List[Any]] signature from Task 1