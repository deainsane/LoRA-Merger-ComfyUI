# Delta-space + `average_weights` for SLERP / NuSLERP / Karcher / NearSwap

**Date:** 2026-07-21
**Status:** Approved (design)
**Scope:** `custom_nodes/LoRA-Merger-ComfyUI`

## Problem

After the GTA rework, the GTA family (linear/task_arithmetic/ties/dare/della/breadcrumbs)
and SCE were moved onto the **delta-space** merge path (materialize each LoRA's full delta →
merge the deltas → refactor back to a LoRA via randomized SVD). They reconstruct the intended
merged delta faithfully.

The remaining non-GTA methods — **SLERP, NuSLERP, Karcher, NearSwap** — are still on the old
**factored path**, which merges the `up` and `down` LoRA factors *separately* and reconstructs
`up @ down`. Merging the factors independently injects meaningless `up_i @ down_j` cross-terms,
so the reconstructed delta only weakly resembles the true merge (~0.67 cosine measured; the
delta-space equivalent measures ~0.87).

Separately, these four have no magnitude/normalization control. SCE and the GTA family expose an
`average_weights` toggle (OFF = additive sum, strengths act as gains, matches ComfyUI's native
stacking, the default; ON = normalized average/blend). These four should expose the same control
for consistency and so per-LoRA strength behaves predictably.

## Goals

1. Migrate SLERP, NuSLERP, Karcher, NearSwap to the delta-space path (fidelity ~0.67 → ~0.87).
2. Add the `average_weights` toggle (default OFF) to all four, with the *same* meaning as every
   other merge node.
3. Keep VRAM discipline consistent with the rest of the delta-space path (no OOM regressions).

## Non-goals

- No change to the GTA family or SCE.
- No new interpolation semantics beyond wiring the magnitude control; blend *position* still comes
  from each method's own parameter (SLERP `t`, NearSwap threshold, NuSLERP/Karcher default even).
- Not preserving relative-strength control of interpolation position (see Design §3).

## Design

### 1. Routing (`src/lora_mergekit_merge.py`, `merge()` / `process_key`)

Extend the existing `is_delta_space` branch to also match
`(not is_clip) and mode in {"slerp", "nuslerp", "karcher", "nearswap"}` (call this set
`INTERP_MODES`). CLIP layers keep using `simple_weighted_average` (the existing `is_clip` path) —
delta-space routing is UNet-only, as it already is for GTA/SCE. The branch
already: builds each owner's full delta `scale·(u @ d)` with `scale = alpha/rank` (reshaping 4-D
conv factors to 2-D), computes a merged dense delta, refactors via
`merged_delta_to_lora(..., energy=refactor_energy)`, applies `output_scale` (lambda) once to `up`,
and returns `(up, down, alpha_out)`.

For `INTERP_MODES` the merged delta is produced by §2 instead of `gta_merge` / `sce_merge_deltas`.

The **CUDA worker-serialization** guard (`n_workers = 1` on CUDA) must also include `INTERP_MODES`,
since these now materialize dense deltas and run rSVD in the refactor. Update the
`is_delta_space` used for `n_workers` (currently `mode in GTA_MODES or mode == "sce"`) to also
include `INTERP_MODES`. Factor the delta-space mode test into one place so the per-key routing and
the worker-count use the same predicate.

### 2. Blend + magnitude

Run the existing method function (`slerp_merge` / `nuslerp_merge` / `karcher_merge` /
`nearswap_merge`, obtained via the already-in-scope `method`) on the **list of deltas** using
**unit weights** (weight = 1.0 for every owner). This yields the blend `B` with ‖B‖ ≈ one delta:
SLERP/Karcher/NearSwap pre-scale their inputs by weight (so weight 1 = pure blend); NuSLERP uses
weight only as a ratio (unit weights → even midpoint). Verified numerically 2026-07-21: unit-weight
blend magnitude ≈ average input-delta magnitude for all four.

A shared helper builds the mergekit plumbing from the delta list and calls the method:

```
def _run_method_on_deltas(method, deltas, key, method_args, device, dtype):
    # deltas: list[Tensor]; returns the merged dense delta.
    tensor_map, weight_map = {}, {}
    weight_info = WeightInfo(name=f"{key}.merge", dtype=dtype, is_embed=False)
    for i, d in enumerate(deltas):
        ref = ModelReference(model=ModelPath(path=f"{key}.{i}"))
        tensor_map[ref] = d
        weight_map[ref] = torch.tensor(1.0)          # unit weight -> pure blend
    gather = GatherTensors(weight_info=create_map(key, tensor_map, dtype))
    params = ImmutableMap({r: ImmutableMap(create_tensor_param(weight_map[r], method_args))
                           for r in tensor_map})
    return method(tensor_map, gather, weight_info, params, method_args)
```

The method's internal `method_args['lambda_']` is 1.0 (set in `prepare_method_args`), so the
method's own lambda multiply is a no-op; global `output_scale` is applied later in the branch.

Then post-scale by the key's per-LoRA strengths `weights` (already gathered as `strength_model`
or `strength_clip` for the key):

```
mag = weights.abs()
strength_scale = float(mag.sum()) if not normalize else float(mag.mean())
merged = B * strength_scale
```

- `average_weights = OFF` (`normalize=False`, default): `merged = B · Σ|sᵢ|` → additive; two LoRAs
  at strength 1 → 2× (stacked LoRAs keep full magnitude; strength is a gain).
- `average_weights = ON` (`normalize=True`): `merged = B · mean(|sᵢ|)` → blend; two LoRAs at
  strength 1 → 1×.

### 3. Consequence: strength = magnitude only, not blend position

Because the blend uses unit weights, relative strengths no longer shift the interpolation
*position* — they control magnitude only. Blend position comes from each method's own control:
SLERP `t`, NearSwap `similarity_threshold`, NuSLERP/Karcher default to the even midpoint/mean.
NuSLERP-with-2-models therefore collapses toward SLERP@t=0.5; users wanting ratio-controlled
position use SLERP's `t`. This is the accepted price of a consistent magnitude story across all
nodes.

### 4. Edge cases

- **Single-owner key** (only one LoRA has the key): skip the method (SLERP/NuSLERP/NearSwap need
  ≥2). Fallback: `merged = delta · strength_scale` where `strength_scale` uses the single owner's
  `|s|` (Σ and mean coincide). This mirrors SCE's single-LoRA guard.
- **Arity constraints**: SLERP and NearSwap require exactly two owners; keep their existing
  behavior (the method raises if the count is unsupported). Karcher accepts N; NuSLERP is 2-model.
  No new validation beyond the single-owner fallback.
- **Refactor / dtype / conv reshape**: identical to the GTA/SCE delta-space branch (deltas built
  as float32; `merged_delta_to_lora` returns a fresh alpha; 4-D conv factors reshaped to 2-D
  before `u@d`, as the existing branch already does).

### 5. Nodes & wiring (`src/nodes_merge_methods.py`)

Add `average_weights` BOOLEAN (default `False`) to `SLERPMergeMethod`, `NuSlerpMergeMethod`,
`KArcherMergeMethod`, `NearSwapMergeMethod`. Each `get_method` adds `"normalize": average_weights`
to its `settings` dict (alongside the existing `t` / `nuslerp_*` / `max_iter`,`tol` /
`similarity_threshold`). Tooltip mirrors the SCE/GTA wording (OFF = additive sum / gain / default;
ON = normalized average / blend).

`prepare_method_args` already merges `method['settings']` into `method_args`, so `normalize`
propagates automatically; the delta-space branch reads `method_args.get('normalize', False)`.

## Testing

Standalone numeric scripts (repo pytest does not collect — known import-path breakage), plus the
standalone widget-name test:

1. **Fidelity**: for each of the four, delta-space cosine to the additive/mean reference is
   materially higher than the factored path (~0.87 vs ~0.67 for SLERP/NuSLERP/Karcher).
2. **Magnitude**: `average_weights=OFF` → ‖out‖ ≈ Σ|s|·‖B‖; `ON` → mean(|s|)·‖B‖. Strength acts
   as a linear gain in additive mode (0.5/1.0/2.0 → linear).
3. **Single-owner fallback** returns the (strength-scaled) delta, not zero.
4. **Node wiring**: extend `tests/test_merge_node_names.py` — each of the four exposes
   `average_weights`, defaults OFF, and wires to internal `normalize`.

## Files touched

- `src/lora_mergekit_merge.py` — routing, blend helper, magnitude post-scale, worker-guard predicate.
- `src/nodes_merge_methods.py` — `average_weights` widget + settings on the four nodes.
- `tests/test_merge_node_names.py` — wiring coverage for the four nodes.
- (No change to `src/merge/gta.py`, `src/merge/algorithms.py` merge cores; the four method
  functions are reused as-is.)
