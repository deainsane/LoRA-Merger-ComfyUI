# Spec: Remove lora_stack from PM Block Selector

## Status
Approved for implementation.

## Problem

`BlockSelector` currently takes `lora_stack` as a required input. This couples it to the `LoRAStack` object, forcing the user to wire the stack into every BlockSelector node. The node only uses the stack to resolve an integer index to a lora name — a pure bookkeeping concern that should not require the full stack.

## Goal

Make `BlockSelector` a self-contained, stack-agnostic node. It outputs `(index, block_definition)` pairs that `LoraDecompose` resolves using its own `key_dicts`.

## Design

### 1. BlockSelector becomes a no-op node

**File:** `src/nodes_block_selector.py`

**Inputs:**
- `block_definition` (BlockDefinition, required) — per-block weight config
- `index` (INT, required) — 0-based position in the LoRA stack that `LoraDecompose` will resolve
- `block_selection` (BlockSelection, optional) — chained output from previous BlockSelector

**Output:** `BlockSelection` dict:
```python
{
    "configs": {index: block_definition, ...},
    "chain": block_selection  # previous BlockSelection or None
}
```

**Logic:**
- `index` must be `>= 0`
- If `block_selection` is provided, check `index not in block_selection["configs"]` — if it already exists, raise `ValueError` with a clear message
- Build and return the new `BlockSelection` dict

**Return type string:** unchanged `("BlockSelection",)` — ComfyUI consumers that only chain will work unchanged.

### 2. New helpers in `src/blocks.py`

**`build_block_selection_dict(chain, index, definition) -> dict`**
- Accepts optional `chain` (the previous `BlockSelection` dict or `None`)
- Validates no index collision
- Returns a new `BlockSelection`-shaped dict with `configs` and `chain` merged

**`resolve_block_selection(selection, lora_names) -> Optional[Dict[str, Dict[str, float]]]`**
- `selection`: the `BlockSelection` dict from BlockSelector
- `lora_names`: ordered list of LoRA names from `LoraDecompose.key_dicts`
- For each `(index, definition)` in `selection["configs"]`, validates index in range, resolves to lora name, and calls `compute_lora_weights(keys_by_name[lora_name], definition)` to produce per-key weight dict
- Returns `None` if no configs or if the resolved dict is empty, else returns the merged `{lora_name: {key: weight}}` dict for use by `apply_block_weights`

### 3. LoraDecompose consumes new selection format

**File:** `src/lora_decompose.py`

**Change:** In `decompose()`, before the per-key loop, call:
```python
resolved_selection = resolve_block_selection(block_selection, list(key_dicts.keys()))
```
Then pass `resolved_selection` (or `None`) to `apply_block_weights()` instead of the raw `block_selection`.

The hash check for `last_block_selection_hash` continues to work because it hashes the `BlockSelection` dict directly (which is unchanged in structure).

### 4. Type definition

**File:** `src/types.py`

Add:
```python
# BlockSelection dict: maps stack index -> BlockDefinition
BlockSelectionConfig = Dict[int, Dict[str, Any]]
```

And update `__all__`.

### 5. Tests

**File:** `tests/test_blocks.py`

- Rewrite `TestApplySelection` to use the new index-based chaining pattern
- Rename tests to reflect new semantics (e.g., `test_selects_by_index`)
- Add `test_index_collision_raises_error` — verify chaining two BlockSelectors with the same index raises `ValueError`
- Add `test_resolve_block_selection` tests covering:
  - happy path (index resolved to lora name, weights computed)
  - out-of-range index passes through (warning log)
  - empty configs returns `None`
  - empty lora_names returns `None`

All other tests in `test_blocks.py` remain unchanged.

### 6. No changes needed elsewhere

- `lora_mergekit_merge.py` — no block selection coupling
- `lora_stack_sampler.py` — no block selection coupling
- `lora_power_stacker.py` — no block selection coupling
- `__init__.py` — `NODE_CLASS_MAPPINGS` unchanged, `BlockSelector` return type string unchanged
- `js/` frontend — no changes (type string unchanged)

## Data Flow Summary

```
LoRAPowerStacker → key_dicts (LoRAStack)
LoRAPowerStacker → strengths (LoRAWeights)
                     ↓
BlockSelector → block_selection (BlockSelection: {configs: {0: defA, ...}, chain: prev})
                     ↓
LoraDecompose → resolves index → lora name via key_dicts
                 computes per-key weights via definition
                 applies via apply_block_weights()
                     ↓
LoraMergerMergekit → merged LoRABundle
```

## Error Handling

| Situation | Behavior |
|-----------|----------|
| Index collision in chain | `ValueError` raised in `build_block_selection_dict` |
| Index out of range in resolve | Warning logged, config skipped, other configs processed |
| Empty block_selection | `None` passed to `apply_block_weights` (no-op, unchanged) |
| Negative index | `ValueError` raised in `BlockSelector.select` |

## Backward Compatibility

Existing workflows that chain BlockSelector nodes will continue to work unchanged — the `BlockSelection` dict structure (`{"configs": {}, "chain": None}`) is identical to the current flat dict format because the current dict is `{lora_name: {key: weight}}` and the new dict is `{configs: {index: definition}, chain: prev}`. These are different shapes, so any node that directly constructs a `BlockSelection` dict (rather than using `BlockSelector`) would need updating, but no such node exists.