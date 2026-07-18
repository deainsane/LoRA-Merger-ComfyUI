# PM Block Selector + Model-Specific Block Nodes — Design

Date: 2026-07-18
Status: Approved (design), pending implementation plan

## Goal

Add per-block, per-LoRA weighting to the LoRA PowerMerge pipeline so a user can scale
individual transformer-block groups of each LoRA in a `LoRAStack` before it is decomposed
and merged. Deliver three new nodes plus a small change to `PM LoRA Stack Decompose`:

- **PM Block Selector** — model-agnostic. Binds a `BlockDefinition` to one LoRA (by stack
  index) and produces/accumulates a `BlockSelection`. Chainable to cover multiple LoRAs.
- **PM KREA 2 Blocks** — model-specific `BlockDefinition` source for KREA2 LoRAs.
- **PM FLUX.2.Klein Blocks** — model-specific `BlockDefinition` source for FLUX.2-Klein LoRAs.

## Key decisions (locked)

1. **Selection semantics:** per-block *strength weights* (float multipliers), not binary
   keep/drop. Weight `0.0` means "drop this block for this LoRA"; `1.0` means unchanged.
2. **Weights live in the model-specific nodes.** The Block Selector is model-agnostic and
   carries no per-block widgets. This keeps the selector reusable across models and needs no
   dynamic-widget JS.
3. **Grouping:** configurable grouped ranges (a `group_size` per block category), like the
   existing DiT `layers_per_group` grouping.
4. **Block count is detected at runtime** from the connected LoRA's keys (both sample LoRAs
   only train a subset of blocks). Model nodes supply only the key *patterns* + grouping +
   weights, never a hardcoded block count.
5. **Weight entry UX:** string weight lists — each block category has a `group_size` INT and a
   comma-separated per-group weight STRING (applied left-to-right; missing groups fall back to
   a default of 1.0). Small named pathways get individual FLOAT widgets. No new JS.
6. **Weight application point:** `PM LoRA Stack Decompose`, applied to the `up` factor only so
   the block weight scales the reconstructed delta *linearly* (consistent with the
   strength-squaring fix: scaling both up and down would square the weight).

## Derived block structure (from real sample LoRAs)

Introspected on 2026-07-18:

**KREA2** — `SarahF.Krea2-LoRA-v01.safetensors` (512 keys, DiT, unified block stack):
- `diffusion_model.blocks.{0..27}` — 28 main transformer blocks
  (`attn.{wq,wk,wv,wo,gate}`, `mlp.{up,gate,down}`)
- `diffusion_model.txtfusion.layerwise_blocks.{0..3}` — 4 blocks
- `diffusion_model.txtfusion.refiner_blocks.{0..3}` — 4 blocks
- `diffusion_model.txtmlp.*` — optional (absent in this LoRA; present in some KREA2 LoRAs)

**FLUX.2-Klein** — `ultra_real_v4.safetensors` (224 keys, Flux dual-stream):
- `diffusion_model.double_blocks.{0..7}` — `img_attn`, `img_mlp`, `txt_attn`, `txt_mlp`
- `diffusion_model.single_blocks.{0..23}` — `linear1`, `linear2`

Both LoRAs train only a subset of the full model's blocks — reinforcing runtime detection.

## Data types

Two new ComfyUI socket types.

### BlockDefinition (model node → Block Selector)

Plain dict; patterns are hardcoded inside each model node.

```python
{
  "model": "KREA2",
  "categories": [                                   # indexed, grouped block stacks
    {"name": "blocks",
     "regex": r"(?:^|\.)blocks\.(\d+)\.",           # capture group 1 = block index
     "group_size": 5,
     "group_weights": [1.0, 1.0, 1.0, 0.8, 0.5, 0.0],
     "default_weight": 1.0}                          # groups beyond the list
  ],
  "pathways": [                                      # single-weight, non-grouped
    {"name": "txtfusion.layerwise", "regex": r"txtfusion\.layerwise_blocks\.", "weight": 1.0},
    {"name": "txtfusion.refiner",   "regex": r"txtfusion\.refiner_blocks\.",   "weight": 1.0},
    {"name": "txtmlp",              "regex": r"(?:^|\.)txtmlp\.",              "weight": 1.0}
  ]
}
```

FLUX.2-Klein definition has two categories (`double_blocks`, `single_blocks`) and no pathways.

### BlockSelection (Block Selector → Block Selector → Decompose)

Accumulated map keyed by LoRA *name* (stable across nodes; names are unique in a stack). Only
non-1.0 weights are stored; missing keys default to 1.0 downstream.

```python
{
  "SarahF.Krea2-LoRA-v01": {"diffusion_model.blocks.15.attn.wq.weight": 0.5, ...},
  "ultra_real_v4":         {"diffusion_model.single_blocks.20.linear1.weight": 0.0, ...}
}
```

## Nodes

### PM KREA 2 Blocks
- Output: `BlockDefinition`.
- Widgets: `blocks_group_size` (INT, default 5), `blocks_weights` (STRING, default `"1.0"`),
  `txtfusion_layerwise` (FLOAT, default 1.0), `txtfusion_refiner` (FLOAT, default 1.0),
  `txtmlp` (FLOAT, default 1.0).
- Builds the KREA2 `BlockDefinition` from the widget values + hardcoded regex patterns.

### PM FLUX.2.Klein Blocks
- Output: `BlockDefinition`.
- Widgets: `double_blocks_group_size` (INT, default 1), `double_blocks_weights` (STRING,
  default `"1.0"`), `single_blocks_group_size` (INT, default 5), `single_blocks_weights`
  (STRING, default `"1.0"`).

### PM Block Selector
- Inputs: `lora_stack` (LoRAStack), `block_definition` (BlockDefinition),
  `block_selection` (BlockSelection, optional).
- Widget: `index` (INT, default 0) — which LoRA in the stack to target.
- Output: `BlockSelection`.
- Logic:
  1. Copy the incoming `block_selection` (or start empty).
  2. Resolve `lora_name = list(lora_stack.keys())[index]` (bounds-checked).
  3. For each layer key of that LoRA (string-normalized, tuple keys handled):
     match categories first (extract block index → `group = idx // group_size` →
     `group_weights[group]` or `default_weight`), then pathways (single weight),
     else 1.0 (unmatched, unchanged). Record only weights != 1.0.
  4. Set `block_selection[lora_name] = per_key_weights` (override on conflict, with a warning).
  5. Return `block_selection`.

### PM LoRA Stack Decompose (changed)
- New optional input `block_selection` (BlockSelection); folded into the existing hash cache
  signature.
- In `process_key`, immediately after `calc_up_down_alphas`: for each owning LoRA,
  `w = block_selection.get(lora_name, {}).get(str(key), 1.0)`.
  - `w == 0` → drop that LoRA from the key (key disappears if it was the only owner).
  - else → scale `up *= w` (linear delta scaling), leave `down`/`alpha` untouched.
  - if all owners are dropped, skip the key entirely.
- Applied before the decomposition-method branch so SVD rank alignment sees the scaled delta.

## Error handling
- `index` out of range or empty stack → warn, return the incoming selection unchanged.
- LoRA with zero matching keys → warn ("definition may not match this LoRA's architecture");
  the entry is effectively a no-op (all weights 1.0).
- Malformed `*_weights` string → warn, treat unparseable entries as `default_weight` (1.0).
- `group_weights` shorter than the detected group count → remainder uses `default_weight`;
  longer → extra values ignored.
- Decompose ignores `block_selection` entries whose LoRA name is not in the stack.

## Files
- **New** `src/blocks.py` — pure logic, no ComfyUI deps, fully unit-testable:
  weight-string parsing, category/pathway matching + grouping, per-key weight computation for
  a LoRA's key set, and BlockSelection merge/override.
- **New** `src/nodes_block_selector.py` — thin ComfyUI node wrappers for the three nodes.
- **Changed** `src/lora_decompose.py` — optional `block_selection` input + `up` scaling.
- **Changed** `__init__.py` — register the three nodes in `NODE_CLASS_MAPPINGS` /
  `NODE_DISPLAY_NAME_MAPPINGS`, category `LoRA PowerMerge`.

## Testing
- Unit: weight-string parsing (empty, partial, extra, whitespace, invalid tokens).
- Unit: KREA2 and Klein detection→weight mapping on synthetic keys (correct group→weight,
  pathway matching, unmatched→1.0).
- Unit: Block Selector chaining — two selectors on different indices accumulate; same index
  overrides; out-of-range index is a safe no-op.
- Unit/numeric: Decompose applies weight to `up` — verify delta scales linearly
  (`0.5 → 0.5×`, `1.0 → unchanged`, `0.0 → key dropped`).
- Integration: synthetic `LoRAStack` through KREA2 Blocks → Block Selector → Decompose;
  assert per-key scaling.
- Verification step (implementation): load the two real sample LoRAs into an actual
  `LoRAStack` and confirm the regex patterns match the ComfyUI-internal layer keys; adjust
  patterns if ComfyUI transforms the `diffusion_model.…blocks.N.…` structure.

## Risks / open implementation notes
- **Layer-key format inside a loaded `LoRAStack`.** The regexes assume keys retain the
  `diffusion_model.…blocks.N.…` structure after `comfy.lora.load_lora`. Patterns use
  `(?:^|\.)…` anchors to tolerate prefix differences, but this must be verified against a real
  loaded stack before finalizing (first implementation task).
- Keying `BlockSelection` by LoRA name assumes unique names (guaranteed by the Power Stacker's
  collision handling) and that Decompose receives the same stack the Selector saw.
