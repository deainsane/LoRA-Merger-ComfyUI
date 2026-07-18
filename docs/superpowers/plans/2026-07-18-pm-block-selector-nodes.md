# PM Block Selector + Model-Specific Block Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-block, per-LoRA weighting to the LoRA PowerMerge pipeline via three new nodes (PM Block Selector, PM KREA 2 Blocks, PM FLUX.2.Klein Blocks) plus a small change to PM LoRA Stack Decompose.

**Architecture:** All logic lives in a new dependency-free `src/blocks.py` (pure functions, unit-testable without ComfyUI). Node classes in `src/nodes_block_selector.py` are thin wrappers. `src/lora_decompose.py` gains an optional `block_selection` input that scales each LoRA's `up` factor per key (linear delta scaling; weight 0 drops the key). Two new socket types: `BlockDefinition` (model node → selector) and `BlockSelection` (selector → selector → decompose).

**Tech Stack:** Python, PyTorch, ComfyUI custom-node API, pytest.

---

## Background the engineer needs

- A `LoRAStack` is `Dict[lora_name, Dict[layer_key, LoRAAdapter]]`. Layer keys are ComfyUI-internal model keys like `"diffusion_model.blocks.0.attn.wq.weight"`. **Some keys are tuples** `("diffusion_model.single_blocks.0.linear1.weight", (0, 0, N))` from qkv-splitting — always normalize with `str(key[0])` before regex.
- `BlockDefinition` dict shape:
  ```python
  {"model": "KREA2",
   "categories": [{"name": "blocks", "regex": r"(?:^|\.)blocks\.(\d+)\.",
                   "group_size": 5, "group_weights": [1.0, 0.5], "default_weight": 1.0}],
   "pathways": [{"name": "txtmlp", "regex": r"(?:^|\.)txtmlp\.", "weight": 1.0}]}
  ```
  A *category* is an indexed block stack (regex capture group 1 = block index → `group = idx // group_size`). A *pathway* is a single-weight group (no index).
- `BlockSelection` dict shape, keyed by LoRA name, storing only non-1.0 weights by normalized key string:
  ```python
  {"SarahF.Krea2-LoRA-v01": {"diffusion_model.blocks.15.attn.wq.weight": 0.5}}
  ```
- Block weight is applied to the **`up`** factor only (linear delta scaling). This mirrors the strength-linearization already in `lora_mergekit_merge.py`; scaling both `up` and `down` would square the weight.

## Running the tests

`src/blocks.py` has **no** ComfyUI or relative-import dependencies, so its tests import cleanly. The repo's broader test suite has a pre-existing collection failure in other files — run the new test file in isolation:

```
/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -v -p no:cacheprovider
```

`tests/conftest.py` already inserts `src/` onto `sys.path`, so `from blocks import ...` resolves.

## File structure

- **Create** `src/blocks.py` — all pure logic: key normalization, weight-string parsing, category/pathway construction & matching, per-LoRA weight computation, selection merge, definition builders, and `up`-scaling.
- **Create** `src/nodes_block_selector.py` — three thin node classes.
- **Create** `tests/test_blocks.py` — unit tests for `src/blocks.py`.
- **Modify** `src/lora_decompose.py` — optional `block_selection` input + cache + `up` scaling.
- **Modify** `__init__.py` — register the three nodes.

---

## Task 1: Key normalization + weight-string parsing

**Files:**
- Create: `src/blocks.py`
- Test: `tests/test_blocks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_blocks.py
from blocks import normalize_key, parse_weight_list


class TestNormalizeKey:
    def test_string_key_unchanged(self):
        assert normalize_key("diffusion_model.blocks.0.attn.wq.weight") == \
            "diffusion_model.blocks.0.attn.wq.weight"

    def test_tuple_key_uses_first_element(self):
        assert normalize_key(("diffusion_model.single_blocks.0.linear1.weight", (0, 0, 9))) == \
            "diffusion_model.single_blocks.0.linear1.weight"


class TestParseWeightList:
    def test_simple_list(self):
        assert parse_weight_list("1,1,0.8,0.5,0") == [1.0, 1.0, 0.8, 0.5, 0.0]

    def test_whitespace_tolerated(self):
        assert parse_weight_list(" 1.0 , 0.5 ") == [1.0, 0.5]

    def test_empty_token_uses_default(self):
        assert parse_weight_list("1,,0.5", default=1.0) == [1.0, 1.0, 0.5]

    def test_empty_string_returns_single_default(self):
        assert parse_weight_list("", default=1.0) == [1.0]

    def test_invalid_token_uses_default(self):
        assert parse_weight_list("1,abc,0.5", default=1.0) == [1.0, 1.0, 0.5]

    def test_none_returns_empty(self):
        assert parse_weight_list(None) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -v -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'blocks'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/blocks.py
"""
Pure logic for block-wise LoRA weighting (PM Block Selector + model block nodes).

No ComfyUI or intra-package imports so this module is unit-testable in isolation.
"""
import logging
import re
from typing import Any, Dict, Iterable, List, Optional


def normalize_key(key: Any) -> str:
    """Return the string form of a LoRA layer key (ComfyUI keys may be tuples)."""
    if isinstance(key, tuple):
        return str(key[0])
    return str(key)


def parse_weight_list(s: Optional[str], default: float = 1.0) -> List[float]:
    """Parse a comma-separated per-group weight string into a list of floats.

    Empty or unparseable tokens fall back to ``default``. ``None`` yields ``[]``.
    """
    if s is None:
        return []
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok == "":
            out.append(default)
            continue
        try:
            out.append(float(tok))
        except ValueError:
            logging.warning(f"[PM Blocks] Could not parse weight '{tok}', using {default}")
            out.append(default)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -v -p no:cacheprovider`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/blocks.py tests/test_blocks.py
git commit -m "feat(blocks): key normalization and weight-string parsing"
```

---

## Task 2: Category construction + per-key weight lookup

**Files:**
- Modify: `src/blocks.py`
- Test: `tests/test_blocks.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_blocks.py
from blocks import make_category, key_weight

KREA2_DEF = {
    "model": "KREA2",
    "categories": [make_category("blocks", r"(?:^|\.)blocks\.(\d+)\.", 5, "1,1,1,0.8,0.5,0")],
    "pathways": [
        {"name": "txtfusion.layerwise", "regex": r"txtfusion\.layerwise_blocks\.", "weight": 0.3},
        {"name": "txtfusion.refiner", "regex": r"txtfusion\.refiner_blocks\.", "weight": 0.7},
        {"name": "txtmlp", "regex": r"(?:^|\.)txtmlp\.", "weight": 0.0},
    ],
}


class TestMakeCategory:
    def test_structure(self):
        cat = make_category("blocks", r"(?:^|\.)blocks\.(\d+)\.", 5, "1,0.5")
        assert cat == {"name": "blocks", "regex": r"(?:^|\.)blocks\.(\d+)\.",
                       "group_size": 5, "group_weights": [1.0, 0.5], "default_weight": 1.0}

    def test_group_size_clamped_to_at_least_one(self):
        assert make_category("blocks", r"x", 0, "1")["group_size"] == 1


class TestKeyWeight:
    def test_block_in_first_group(self):
        # blocks 0-4 -> group 0 -> 1.0
        assert key_weight("diffusion_model.blocks.3.attn.wq.weight", KREA2_DEF) == 1.0

    def test_block_in_fourth_group(self):
        # blocks 15-19 -> group 3 -> 0.8
        assert key_weight("diffusion_model.blocks.17.mlp.up.weight", KREA2_DEF) == 0.8

    def test_block_beyond_weight_list_uses_default(self):
        # block 40 -> group 8, list has 6 entries -> default_weight 1.0
        assert key_weight("diffusion_model.blocks.40.attn.wq.weight", KREA2_DEF) == 1.0

    def test_category_regex_does_not_match_double_blocks(self):
        # 'blocks' category must NOT catch 'double_blocks' (preceded by '_')
        assert key_weight("diffusion_model.double_blocks.2.img_attn.qkv.weight", KREA2_DEF) == 1.0

    def test_pathway_layerwise(self):
        assert key_weight("diffusion_model.txtfusion.layerwise_blocks.1.attn.wq.weight", KREA2_DEF) == 0.3

    def test_pathway_refiner(self):
        assert key_weight("diffusion_model.txtfusion.refiner_blocks.0.mlp.up.weight", KREA2_DEF) == 0.7

    def test_pathway_txtmlp(self):
        assert key_weight("diffusion_model.txtmlp.0.weight", KREA2_DEF) == 0.0

    def test_unmatched_key_is_one(self):
        assert key_weight("diffusion_model.final_layer.weight", KREA2_DEF) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -k "MakeCategory or KeyWeight" -v -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'make_category'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/blocks.py

def make_category(name: str, regex: str, group_size: int,
                  weights_str: str, default_weight: float = 1.0) -> Dict[str, Any]:
    """Build a category dict (indexed block stack) for a BlockDefinition."""
    return {
        "name": name,
        "regex": regex,
        "group_size": max(1, int(group_size)),
        "group_weights": parse_weight_list(weights_str, default_weight),
        "default_weight": default_weight,
    }


def key_weight(key_str: str, definition: Dict[str, Any]) -> float:
    """Resolve the weight for a single normalized layer key against a BlockDefinition.

    Categories are matched first (index -> group -> weight), then pathways.
    Unmatched keys return 1.0 (unchanged).
    """
    for cat in definition.get("categories", []):
        m = re.search(cat["regex"], key_str)
        if m:
            idx = int(m.group(1))
            group = idx // cat["group_size"]
            gw = cat["group_weights"]
            return gw[group] if group < len(gw) else cat.get("default_weight", 1.0)
    for pathway in definition.get("pathways", []):
        if re.search(pathway["regex"], key_str):
            return pathway["weight"]
    return 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -v -p no:cacheprovider`
Expected: PASS (all Task 1 + Task 2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/blocks.py tests/test_blocks.py
git commit -m "feat(blocks): category construction and per-key weight lookup"
```

---

## Task 3: Per-LoRA weight computation, selection merge, and index selection

**Files:**
- Modify: `src/blocks.py`
- Test: `tests/test_blocks.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_blocks.py
from collections import OrderedDict
from blocks import compute_lora_weights, merge_selection, apply_selection


class TestComputeLoraWeights:
    def test_only_non_default_weights_stored(self):
        keys = [
            "diffusion_model.blocks.0.attn.wq.weight",    # group 0 -> 1.0 (omitted)
            "diffusion_model.blocks.25.attn.wq.weight",   # group 5 -> 0.0 (stored)
            "diffusion_model.txtmlp.0.weight",            # pathway -> 0.0 (stored)
        ]
        out = compute_lora_weights(keys, KREA2_DEF)
        assert out == {
            "diffusion_model.blocks.25.attn.wq.weight": 0.0,
            "diffusion_model.txtmlp.0.weight": 0.0,
        }

    def test_tuple_keys_normalized(self):
        keys = [("diffusion_model.blocks.25.attn.wq.weight", (0, 0, 9))]
        out = compute_lora_weights(keys, KREA2_DEF)
        assert out == {"diffusion_model.blocks.25.attn.wq.weight": 0.0}


class TestMergeSelection:
    def test_adds_new_lora(self):
        out = merge_selection({"a": {"k": 0.5}}, "b", {"k2": 0.2})
        assert out == {"a": {"k": 0.5}, "b": {"k2": 0.2}}

    def test_override_existing_lora(self):
        out = merge_selection({"a": {"k": 0.5}}, "a", {"k2": 0.2})
        assert out == {"a": {"k2": 0.2}}

    def test_does_not_mutate_input(self):
        base = {"a": {"k": 0.5}}
        merge_selection(base, "b", {"k2": 0.2})
        assert base == {"a": {"k": 0.5}}


class TestApplySelection:
    def _stack(self):
        return OrderedDict([
            ("lora0", ["diffusion_model.blocks.25.attn.wq.weight"]),
            ("lora1", ["diffusion_model.blocks.3.attn.wq.weight"]),
        ])

    def test_selects_by_index(self):
        out = apply_selection(self._stack(), KREA2_DEF, index=0)
        assert out == {"lora0": {"diffusion_model.blocks.25.attn.wq.weight": 0.0}}

    def test_chaining_accumulates_different_indices(self):
        first = apply_selection(self._stack(), KREA2_DEF, index=0)
        second = apply_selection(self._stack(), KREA2_DEF, index=1, incoming_selection=first)
        assert set(second.keys()) == {"lora0", "lora1"}

    def test_out_of_range_index_passes_through(self):
        incoming = {"lora0": {"k": 0.5}}
        out = apply_selection(self._stack(), KREA2_DEF, index=9, incoming_selection=incoming)
        assert out == {"lora0": {"k": 0.5}}

    def test_empty_stack_passes_through(self):
        out = apply_selection(OrderedDict(), KREA2_DEF, index=0, incoming_selection={"x": {}})
        assert out == {"x": {}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -k "ComputeLoraWeights or MergeSelection or ApplySelection" -v -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'compute_lora_weights'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/blocks.py

def compute_lora_weights(keys: Iterable[Any], definition: Dict[str, Any]) -> Dict[str, float]:
    """Map each of a LoRA's layer keys to its block weight, storing only non-1.0 values."""
    weights: Dict[str, float] = {}
    for key in keys:
        key_str = normalize_key(key)
        w = key_weight(key_str, definition)
        if w != 1.0:
            weights[key_str] = w
    return weights


def merge_selection(base: Optional[Dict[str, Dict[str, float]]],
                    lora_name: str,
                    weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Return a copy of ``base`` with ``lora_name`` set to ``weights`` (override on conflict)."""
    out = {name: dict(w) for name, w in (base or {}).items()}
    if lora_name in out:
        logging.warning(f"[PM Block Selector] Overriding existing block selection for '{lora_name}'")
    out[lora_name] = weights
    return out


def apply_selection(keys_by_name: "Dict[str, Iterable[Any]]",
                    definition: Dict[str, Any],
                    index: int,
                    incoming_selection: Optional[Dict[str, Dict[str, float]]] = None
                    ) -> Dict[str, Dict[str, float]]:
    """Compute and merge block weights for the LoRA at ``index`` in an ordered stack.

    ``keys_by_name`` must be insertion-ordered (dict/OrderedDict) so ``index`` is stable.
    Out-of-range index or empty stack passes the incoming selection through unchanged.
    """
    names = list(keys_by_name.keys())
    if not names:
        logging.warning("[PM Block Selector] Empty LoRAStack; passing selection through.")
        return {name: dict(w) for name, w in (incoming_selection or {}).items()}
    if index < 0 or index >= len(names):
        logging.warning(
            f"[PM Block Selector] index {index} out of range (0..{len(names) - 1}); passing through.")
        return {name: dict(w) for name, w in (incoming_selection or {}).items()}
    lora_name = names[index]
    weights = compute_lora_weights(keys_by_name[lora_name], definition)
    if not weights:
        logging.warning(
            f"[PM Block Selector] No block-weight overrides for '{lora_name}' "
            f"(all effective weights are 1.0, or the definition does not match this LoRA).")
    return merge_selection(incoming_selection, lora_name, weights)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -v -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/blocks.py tests/test_blocks.py
git commit -m "feat(blocks): per-LoRA weight computation, selection merge, index selection"
```

---

## Task 4: Apply block weights to (up, down, alpha) tuples

**Files:**
- Modify: `src/blocks.py`
- Test: `tests/test_blocks.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_blocks.py
import torch
from blocks import apply_block_weights


class TestApplyBlockWeights:
    def _uda(self):
        up = torch.ones(4, 2)
        down = torch.ones(2, 3)
        return {"loraA": (up, down, torch.tensor(2.0))}

    def test_no_selection_returns_input(self):
        uda = self._uda()
        assert apply_block_weights(uda, "k", None) is uda

    def test_weight_scales_up_only(self):
        out = apply_block_weights(self._uda(), "k", {"loraA": {"k": 0.5}})
        up, down, alpha = out["loraA"]
        assert torch.allclose(up, torch.full((4, 2), 0.5))
        assert torch.allclose(down, torch.ones(2, 3))   # down untouched
        assert float(alpha) == 2.0

    def test_delta_scales_linearly(self):
        base = self._uda()["loraA"]
        base_delta = base[0] @ base[1]
        up, down, _ = apply_block_weights(self._uda(), "k", {"loraA": {"k": 0.5}})["loraA"]
        assert torch.allclose(up @ down, 0.5 * base_delta)

    def test_weight_zero_drops_lora(self):
        out = apply_block_weights(self._uda(), "k", {"loraA": {"k": 0.0}})
        assert out == {}

    def test_missing_key_defaults_to_one(self):
        out = apply_block_weights(self._uda(), "other_key", {"loraA": {"k": 0.5}})
        up, _, _ = out["loraA"]
        assert torch.allclose(up, torch.ones(4, 2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -k ApplyBlockWeights -v -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'apply_block_weights'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/blocks.py

def apply_block_weights(uda: Dict[str, Any],
                        key_str: str,
                        block_selection: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Scale each LoRA's ``up`` factor by its block weight for ``key_str``.

    ``uda`` maps lora_name -> (up, down, alpha). Weight 0 drops that LoRA from the key.
    Returns the same object when there is no selection to apply.
    """
    if not block_selection:
        return uda
    out: Dict[str, Any] = {}
    for lora_name, (up, down, alpha) in uda.items():
        w = block_selection.get(lora_name, {}).get(key_str, 1.0)
        if w == 0:
            continue
        if w != 1.0:
            up = up * w
        out[lora_name] = (up, down, alpha)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -v -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/blocks.py tests/test_blocks.py
git commit -m "feat(blocks): apply per-block weights to up factor (linear delta scaling)"
```

---

## Task 5: KREA2 and FLUX.2-Klein definition builders

**Files:**
- Modify: `src/blocks.py`
- Test: `tests/test_blocks.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_blocks.py
from blocks import build_krea2_definition, build_klein_definition


class TestBuildKrea2:
    def test_shape_and_matching(self):
        d = build_krea2_definition(blocks_group_size=5, blocks_weights="1,1,1,0.8,0.5,0",
                                   txtfusion_layerwise=0.3, txtfusion_refiner=0.7, txtmlp=0.0)
        assert d["model"] == "KREA2"
        assert len(d["categories"]) == 1
        assert {p["name"] for p in d["pathways"]} == {
            "txtfusion.layerwise", "txtfusion.refiner", "txtmlp"}
        # end-to-end sanity through key_weight
        assert key_weight("diffusion_model.blocks.17.attn.wq.weight", d) == 0.8
        assert key_weight("diffusion_model.txtfusion.refiner_blocks.0.mlp.up.weight", d) == 0.7


class TestBuildKlein:
    def test_shape_and_matching(self):
        d = build_klein_definition(double_blocks_group_size=1, double_blocks_weights="1,0.5",
                                   single_blocks_group_size=5, single_blocks_weights="1,1,0")
        assert d["model"] == "FLUX.2-Klein"
        assert {c["name"] for c in d["categories"]} == {"double_blocks", "single_blocks"}
        assert d["pathways"] == []
        # double_blocks group_size 1 -> block 1 is group 1 -> 0.5
        assert key_weight("diffusion_model.double_blocks.1.img_attn.qkv.weight", d) == 0.5
        # single_blocks group_size 5 -> block 12 is group 2 -> 0.0
        assert key_weight("diffusion_model.single_blocks.12.linear1.weight", d) == 0.0
        # 'single_blocks' must not be matched by a 'double_blocks' regex and vice-versa
        assert key_weight("diffusion_model.single_blocks.0.linear1.weight", d) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -k "BuildKrea2 or BuildKlein" -v -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'build_krea2_definition'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/blocks.py

def build_krea2_definition(blocks_group_size: int, blocks_weights: str,
                           txtfusion_layerwise: float, txtfusion_refiner: float,
                           txtmlp: float) -> Dict[str, Any]:
    """BlockDefinition for KREA2 LoRAs (unified diffusion_model.blocks.N stack + txtfusion/txtmlp)."""
    return {
        "model": "KREA2",
        "categories": [
            make_category("blocks", r"(?:^|\.)blocks\.(\d+)\.", blocks_group_size, blocks_weights),
        ],
        "pathways": [
            {"name": "txtfusion.layerwise", "regex": r"txtfusion\.layerwise_blocks\.",
             "weight": txtfusion_layerwise},
            {"name": "txtfusion.refiner", "regex": r"txtfusion\.refiner_blocks\.",
             "weight": txtfusion_refiner},
            {"name": "txtmlp", "regex": r"(?:^|\.)txtmlp\.", "weight": txtmlp},
        ],
    }


def build_klein_definition(double_blocks_group_size: int, double_blocks_weights: str,
                           single_blocks_group_size: int, single_blocks_weights: str) -> Dict[str, Any]:
    """BlockDefinition for FLUX.2-Klein LoRAs (double_blocks + single_blocks streams)."""
    return {
        "model": "FLUX.2-Klein",
        "categories": [
            make_category("double_blocks", r"(?:^|\.)double_blocks\.(\d+)\.",
                          double_blocks_group_size, double_blocks_weights),
            make_category("single_blocks", r"(?:^|\.)single_blocks\.(\d+)\.",
                          single_blocks_group_size, single_blocks_weights),
        ],
        "pathways": [],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -v -p no:cacheprovider`
Expected: PASS (entire file)

- [ ] **Step 5: Commit**

```bash
git add src/blocks.py tests/test_blocks.py
git commit -m "feat(blocks): KREA2 and FLUX.2-Klein definition builders"
```

---

## Task 6: Node wrapper classes

**Files:**
- Create: `src/nodes_block_selector.py`

- [ ] **Step 1: Write the implementation**

(These are thin ComfyUI wrappers over the tested `blocks` functions; ComfyUI is not available under pytest, so there is no separate unit test — Task 8's ComfyUI load is the verification.)

```python
# src/nodes_block_selector.py
import logging

from .blocks import apply_selection, build_klein_definition, build_krea2_definition

CATEGORY = "LoRA PowerMerge"
_FLOAT = {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}


class KREA2Blocks:
    """Model-specific block definition for KREA2 LoRAs."""
    RETURN_TYPES = ("BlockDefinition",)
    RETURN_NAMES = ("block_definition",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Per-block weight definition for KREA2 LoRAs "
                   "(diffusion_model.blocks.N + txtfusion + txtmlp). "
                   "Connect to PM Block Selector.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "blocks_group_size": ("INT", {"default": 5, "min": 1, "max": 128}),
            "blocks_weights": ("STRING", {"default": "1.0", "multiline": False,
                "tooltip": "Comma-separated per-group weights, left-to-right. "
                           "Missing groups default to 1.0."}),
            "txtfusion_layerwise": ("FLOAT", _FLOAT),
            "txtfusion_refiner": ("FLOAT", _FLOAT),
            "txtmlp": ("FLOAT", _FLOAT),
        }}

    def build(self, blocks_group_size, blocks_weights, txtfusion_layerwise,
              txtfusion_refiner, txtmlp):
        return (build_krea2_definition(blocks_group_size, blocks_weights,
                                       txtfusion_layerwise, txtfusion_refiner, txtmlp),)


class FluxKleinBlocks:
    """Model-specific block definition for FLUX.2-Klein LoRAs."""
    RETURN_TYPES = ("BlockDefinition",)
    RETURN_NAMES = ("block_definition",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Per-block weight definition for FLUX.2-Klein LoRAs "
                   "(double_blocks + single_blocks). Connect to PM Block Selector.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "double_blocks_group_size": ("INT", {"default": 1, "min": 1, "max": 128}),
            "double_blocks_weights": ("STRING", {"default": "1.0", "multiline": False,
                "tooltip": "Comma-separated per-group weights for double_blocks."}),
            "single_blocks_group_size": ("INT", {"default": 5, "min": 1, "max": 128}),
            "single_blocks_weights": ("STRING", {"default": "1.0", "multiline": False,
                "tooltip": "Comma-separated per-group weights for single_blocks."}),
        }}

    def build(self, double_blocks_group_size, double_blocks_weights,
              single_blocks_group_size, single_blocks_weights):
        return (build_klein_definition(double_blocks_group_size, double_blocks_weights,
                                       single_blocks_group_size, single_blocks_weights),)


class BlockSelector:
    """Bind a BlockDefinition to one LoRA (by stack index); chain to cover multiple LoRAs."""
    RETURN_TYPES = ("BlockSelection",)
    RETURN_NAMES = ("block_selection",)
    FUNCTION = "select"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Selects per-block weights for the LoRA at 'index' in the LoRAStack using "
                   "the given BlockDefinition. Chain block_selection outputs to weight multiple "
                   "LoRAs. Feed the result into PM LoRA Stack Decompose.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LoRAStack",),
                "block_definition": ("BlockDefinition",),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000,
                                  "tooltip": "Which LoRA in the stack (0-based) to weight."}),
            },
            "optional": {
                "block_selection": ("BlockSelection",),
            },
        }

    def select(self, lora_stack, block_definition, index, block_selection=None):
        keys_by_name = {name: list(key_dict.keys()) for name, key_dict in lora_stack.items()}
        result = apply_selection(keys_by_name, block_definition, index, block_selection)
        logging.info(f"[PM Block Selector] index {index}: selection now covers "
                     f"{len(result)} LoRA(s).")
        return (result,)
```

- [ ] **Step 2: Byte-compile to catch syntax errors**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m py_compile src/nodes_block_selector.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/nodes_block_selector.py
git commit -m "feat: PM Block Selector, PM KREA 2 Blocks, PM FLUX.2.Klein Blocks nodes"
```

---

## Task 7: Wire block_selection into PM LoRA Stack Decompose

**Files:**
- Modify: `src/lora_decompose.py`

- [ ] **Step 1: Add the import**

At the top of `src/lora_decompose.py`, below `from .utility import map_device, adjust_tensor_dims`, add:

```python
from .blocks import apply_block_weights, normalize_key
```

- [ ] **Step 2: Track the block_selection cache key in `__init__`**

In `LoraDecompose.__init__`, after `self.last_layer_filter: Optional[Set[str]] = None`, add:

```python
        self.last_block_selection_hash: Optional[str] = None
```

- [ ] **Step 3: Add the optional input**

In `INPUT_TYPES`, add an `"optional"` block after the `"required"` dict:

```python
            "optional": {
                "block_selection": ("BlockSelection", {
                    "tooltip": "Per-block, per-LoRA weights from PM Block Selector. "
                               "Scales each LoRA's contribution per block; weight 0 drops the block."}),
            },
```

- [ ] **Step 4: Thread `block_selection` through `lora_decompose` (signature + cache + call)**

Replace the method signature:

```python
    def lora_decompose(self, key_dicts: LORA_STACK = None,
                       decomposition_method="rSVD", svd_rank=-1, device=None):
```

with:

```python
    def lora_decompose(self, key_dicts: LORA_STACK = None,
                       decomposition_method="rSVD", svd_rank=-1, device=None,
                       block_selection=None):
```

Replace the cache-hit condition:

```python
        if (self.last_lora_names_hash == lora_names_hash_new
                and self.last_svd_rank == svd_rank
                and self.last_decomposition_method == decomposition_method
                and self.last_tensor_sum == self.compute_sum(key_dicts)):
```

with:

```python
        if (self.last_lora_names_hash == lora_names_hash_new
                and self.last_svd_rank == svd_rank
                and self.last_decomposition_method == decomposition_method
                and self.last_block_selection_hash == self.compute_hash(block_selection)
                and self.last_tensor_sum == self.compute_sum(key_dicts)):
```

After the line `self.last_decomposition_method = decomposition_method` (in the recompute branch), add:

```python
        self.last_block_selection_hash = self.compute_hash(block_selection)
```

Replace the `self.decompose(...)` call:

```python
        self.last_result = self.decompose(key_dicts=key_dicts, device=device,
                                          decomposition_method=decomposition_method,
                                          svd_rank=svd_rank)
```

with:

```python
        self.last_result = self.decompose(key_dicts=key_dicts, device=device,
                                          decomposition_method=decomposition_method,
                                          svd_rank=svd_rank,
                                          block_selection=block_selection)
```

- [ ] **Step 5: Apply weights inside `decompose` / `process_key`**

Replace the `decompose` signature:

```python
    def decompose(self, key_dicts, device, decomposition_method, svd_rank) -> LORA_TENSORS_BY_LAYER:
```

with:

```python
    def decompose(self, key_dicts, device, decomposition_method, svd_rank,
                  block_selection=None) -> LORA_TENSORS_BY_LAYER:
```

Replace the entire `process_key` function body:

```python
        def process_key(key, device_=device) -> LORA_TENSOR_DICT:
            uda = calc_up_down_alphas(key_dicts, key, load_device=device_, scale_to_alpha_0=True)

            # Determine if SVD should be applied
            if decomposition_method == "none":
                # Check if all LoRAs have the same rank
                ranks = [up.shape[1] for up, _, _ in uda.values()]
                if len(set(ranks)) > 1:
                    rank_info = {lora_name: up.shape[1] for lora_name, (up, _, _) in uda.items()}
                    raise ValueError(
                        f"LoRAs have different ranks for key '{key}': {rank_info}. "
                        f"Please select a decomposition method (SVD, rSVD, or energy_rSVD) to align dimensions."
                    )
                # No adjustment needed
                return uda
            else:
                # Apply the selected decomposition method
                uda_adjusted = adjust_tensor_dims(
                    uda,
                    apply_svd=True,
                    svd_rank=svd_rank,
                    method=decomposition_method
                )
                return uda_adjusted
```

with:

```python
        def process_key(key, device_=device):
            uda = calc_up_down_alphas(key_dicts, key, load_device=device_, scale_to_alpha_0=True)

            # Apply per-block, per-LoRA weights (scales the up factor; weight 0 drops the LoRA)
            uda = apply_block_weights(uda, normalize_key(key), block_selection)
            if not uda:
                return None  # every contribution dropped for this key

            # Determine if SVD should be applied
            if decomposition_method == "none":
                # Check if all LoRAs have the same rank
                ranks = [up.shape[1] for up, _, _ in uda.values()]
                if len(set(ranks)) > 1:
                    rank_info = {lora_name: up.shape[1] for lora_name, (up, _, _) in uda.items()}
                    raise ValueError(
                        f"LoRAs have different ranks for key '{key}': {rank_info}. "
                        f"Please select a decomposition method (SVD, rSVD, or energy_rSVD) to align dimensions."
                    )
                # No adjustment needed
                return uda
            else:
                # Apply the selected decomposition method
                uda_adjusted = adjust_tensor_dims(
                    uda,
                    apply_svd=True,
                    svd_rank=svd_rank,
                    method=decomposition_method
                )
                return uda_adjusted
```

- [ ] **Step 6: Skip dropped keys in the output loop**

Replace:

```python
        for i, key in enumerate(keys):
            out[key] = process_key(key)
            if (i + 1) % update_frequency == 0 or (i + 1) == len(keys):
```

with:

```python
        for i, key in enumerate(keys):
            result = process_key(key)
            if result is not None:
                out[key] = result
            if (i + 1) % update_frequency == 0 or (i + 1) == len(keys):
```

- [ ] **Step 7: Byte-compile**

Run: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m py_compile src/lora_decompose.py && echo OK`
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add src/lora_decompose.py
git commit -m "feat(decompose): optional block_selection input applies per-block weights"
```

---

## Task 8: Register nodes and verify ComfyUI load

**Files:**
- Modify: `__init__.py`

- [ ] **Step 1: Add the import**

In `__init__.py`, after `from .src.lora_power_stacker import LoraPowerStacker`, add:

```python
from .src.nodes_block_selector import BlockSelector, KREA2Blocks, FluxKleinBlocks
```

- [ ] **Step 2: Register class mappings**

In `NODE_CLASS_MAPPINGS`, after the line `"PM LoRA Stack Decompose": LoraDecompose,`, add:

```python
    "PM Block Selector": BlockSelector,
    "PM KREA 2 Blocks": KREA2Blocks,
    "PM FLUX.2.Klein Blocks": FluxKleinBlocks,
```

- [ ] **Step 3: Register display-name mappings**

In `NODE_DISPLAY_NAME_MAPPINGS`, after the line `"PM LoRA Stack Decompose": "PM LoRA Stack Decompose",`, add:

```python
    "PM Block Selector": "PM Block Selector",
    "PM KREA 2 Blocks": "PM KREA 2 Blocks",
    "PM FLUX.2.Klein Blocks": "PM FLUX.2.Klein Blocks",
```

- [ ] **Step 4: Verify the package imports (module-load smoke test)**

Run:
```
cd /home/lars/SD/Apps/ComfyUI && /home/lars/SD/Apps/ComfyUI/.venv/bin/python -c "import importlib; m = importlib.import_module('custom_nodes.LoRA-Merger-ComfyUI'.replace('-', '_'))" 2>/dev/null || echo "expected: hyphenated dir not importable this way — verify in ComfyUI instead"
```

Because the directory name contains hyphens it is not importable as a normal module; the authoritative check is the next step (start ComfyUI) or the direct-import fallback:

```
/home/lars/SD/Apps/ComfyUI/.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); import nodes_block_selector as n; print(n.KREA2Blocks.INPUT_TYPES()); print(n.FluxKleinBlocks.INPUT_TYPES()); print(n.BlockSelector.INPUT_TYPES())"
```
Expected: three INPUT_TYPES dicts print without error.

- [ ] **Step 5: Verify in ComfyUI (real load + patterns against real LoRAs)**

Start ComfyUI (or ask the user to) and confirm the three nodes appear under **LoRA PowerMerge** and the console shows no load error for `LoRA-Merger-ComfyUI`. Then confirm the regex patterns match a real loaded stack by building a tiny graph:
`PM LoRA Power Stacker` (add the KREA2 sample LoRA) → `PM Block Selector` (index 0, KREA2 Blocks with `blocks_weights="0"`) → `PM LoRA Stack Decompose`, and check the log line `[PM Block Selector] index 0: selection now covers 1 LoRA(s).` reports a non-empty selection. Repeat with the FLUX.2-Klein sample LoRA + `PM FLUX.2.Klein Blocks`.

If the selection is empty for a real LoRA, dump the actual stack keys and adjust the category/pathway regexes in `src/blocks.py` (Task 5 builders) accordingly, then re-run the Task 5 tests.

- [ ] **Step 6: Commit**

```bash
git add __init__.py
git commit -m "feat: register PM Block Selector and model block nodes"
```

---

## Self-review notes (for the implementer)

- Full-file test run at the end: `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/test_blocks.py -v -p no:cacheprovider` should be all green.
- The only runtime risk is the exact loaded-stack key format; Task 8 Step 5 is the explicit verification-and-adjust gate for it.
- Do not change `lora_mergekit_merge.py` — block weights are applied entirely at decompose time.
