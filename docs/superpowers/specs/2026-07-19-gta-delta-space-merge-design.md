# Delta-space GTA merge (own TIES/DARE/DELLA), with SVD refactor

**Date:** 2026-07-19
**Status:** Implemented (2026-07-19)
**Area:** `custom_nodes/LoRA-Merger-ComfyUI`

**Post-implementation deviation (2026-07-19):** the merged-delta refactor uses **randomized
SVD** (`torch.svd_lowrank`) with energy-based dynamic rank, **not** a full `torch.linalg.svd`
via `perform_lora_svd`. Full SVD across the 8-worker merge pool under a resident diffusion
model is the fragile, heavy path — ~190 MB workspace and ~2.9 s per 3072² layer through the
cuSOLVER driver — and OOMs/segfaults under VRAM pressure. The randomized path computes only the
top-r components: ~6 MB, ~4 ms, pure matmuls. The refactor algorithm is user-selectable via a
new `refactor_method` widget on the merge node (`energy_rSVD` default, `rSVD`; full SVD
intentionally not offered). See `src/merge/lora_refactor.py`.

## Problem

The `normalize` option on the GTA-family merge nodes (TIES, DARE, DELLA, Breadcrumbs,
Task Arithmetic, Linear — the "PM Ties (mergekit)" node and siblings) almost entirely
destroys the effect strength of the merged LoRAs, worse the more LoRAs are merged.

Root cause: the node merges the `up` and `down` LoRA factors **separately** (via mergekit),
then reconstructs the delta as `up @ down`. Two consequences, both specific to running a
delta-space merge method on the *factors*:

1. **The normalize divisor squares.** mergekit's `normalize` divides each merged factor by
   the per-element sum of contributing weights `D` (`generalized_task_arithmetic.py:171,178`).
   Because it lands on *both* factors, a LoRA's own contribution in the reconstructed delta is
   divided by `D_up · D_down ≈ (Σ_j √|s_j|)²`. For `N` comparable LoRAs that is `≈ N²·s̄`, so
   each LoRA's contribution is suppressed by `≈ 1/N²` instead of the intended `1/N`
   (2 LoRAs → ¼ not ½; 3 → 1/9; 4 → 1/16). This is the "kills almost entirely" symptom.
2. **The TIES sign vote is meaningless on factors.** TIES elects a per-element sign by
   majority, but the sign of an `up`-factor element bears no relation to the sign of the
   delta `up @ down` — the matrix product mixes signs. So "per-element TIES on the factors"
   is not TIES on the LoRA at all.

Both problems share one root: **per-element GTA operations (sign consensus, per-element
normalize, sparsification) are defined on the reconstructed delta and cannot be expressed in
factor space.** No scalar or per-factor correction fixes it; the factor split must be
undone before the merge.

Related prior work: `lora-merger-factored-strength-squaring` memory — the `√s` factor-weight
trick that linearizes strength for the *factored* path. That trick is a workaround for
factor-space and is **not needed** on the delta-space path.

## Decision summary

- **Route the entire GTA family through delta-space**, including linear and task_arithmetic
  (chosen for uniformity even though pure linear does not strictly need it).
- **Write our own GTA implementation** in plain torch; drop the mergekit dependency for this
  family (removes the zeros-base `ModelReference` / `GatherTensors` / `ImmutableMap` adapter
  ceremony). mergekit stays only for the exotic methods (slerp, karcher, nuslerp, sce,
  nearswap), which are out of scope here.
- **Refactor the merged delta back to a LoRA** using the existing SVD framework
  (`perform_lora_svd`, `utility.py:108`), with **energy-based dynamic rank wired in from the
  start** (`sv_cumulative` / `sv_fro`).

## Architecture

### New module: `src/merge/gta.py`

Pure-torch GTA on a list of full delta tensors. No mergekit imports. Public entry point:

```python
def gta_merge(
    deltas: list[Tensor],          # per-LoRA natural deltas (see "delta convention")
    weights: Tensor,               # per-LoRA merge weight = strength_i (signed)
    *,
    mode: str,                     # linear | task_arithmetic | ties | dare | della | breadcrumbs
    sign_consensus: bool,          # elect per-element majority sign then disjoint-merge
    density: float = 1.0,
    epsilon: float = 0.0,          # della
    gamma: float = 0.0,            # breadcrumbs
    normalize: bool = True,
    rescale_norm: str = "default", # default|none|l1|l2|linf
) -> Tensor:                       # merged full delta
```

Internal helpers (each independently testable):

- `sparsify(delta, mode, density, epsilon, gamma) -> mask/rescaled delta`
  - `linear`, `task_arithmetic`: no-op (density treated as 1).
  - `ties`: keep top-`density` fraction by |magnitude| per tensor (trim).
  - `dare`: random Bernoulli keep with prob `density`, rescale survivors by `1/density`.
  - `della`: per-row magnitude-ranked keep probability in `[density-epsilon, density+epsilon]`,
    Bernoulli sample, rescale survivors by `1/p`.
  - `breadcrumbs`: drop the largest `gamma` fraction **and** the smallest `1-density-gamma`
    fraction, keep the middle band (respect the `max(0, 1-density-gamma)` edge case).
- `elect_sign(weighted_deltas) -> sign_mask` — per-element majority sign from the weighted
  sum (TIES "sum" method). Only used when `sign_consensus`.
- `disjoint_merge(weighted_deltas, weights, sign_mask, normalize)`:
  - `mixed = Σ_i (weighted_delta_i · agree_i)` where `agree_i` = element matches elected sign
    (or all-ones when `sign_consensus` is False).
  - `divisor = Σ_i (weight_i · agree_i)`, per element; zeros → 1.
  - `if normalize: mixed /= divisor`.
- `apply_rescale_norm(mixed, deltas, weights, kind)` — optional final renorm that restores the
  merged delta's L1/L2/Linf norm toward the weighted input norm (compensates magnitude lost to
  sparsification/masking), mirroring mergekit `RescaleNorm`. `default` resolves per mode.

**mode → (sign_consensus, sparsify) mapping** (matches the node UI in `nodes_merge_methods.py`):

| mode            | sparsify        | sign_consensus source                    |
|-----------------|-----------------|------------------------------------------|
| linear          | none            | False                                    |
| task_arithmetic | none (density1) | False                                    |
| ties            | ties trim       | always True                              |
| dare            | dare random     | `ties` bool (dare_ties vs dare_linear)   |
| della           | della prob      | `ties` bool (della vs della_linear)      |
| breadcrumbs     | breadcrumbs     | `ties` bool                              |

### Merge flow change: `src/lora_mergekit_merge.py` `process_key`

For UNet keys whose method is in the GTA family (all six modes), replace the per-factor
`calculate()` calls with:

1. **Materialize natural deltas** per LoRA:
   `delta_i = (alpha_i / rank_i) · (up_i @ down_i)` — the LoRA's own delta, **no strength**.
   `rank_i = up_i.shape[1]`. Handle conv (4D) by the same reshape `perform_lora_svd` uses.
   (Confirm the exact alpha/rank scale convention against the downstream apply path during
   implementation; the materialized delta must equal what ComfyUI would apply at strength 1.)
2. **Merge:** `merged = gta_merge(deltas, weights=strengths, mode=…, …)`.
   Strength enters as the per-LoRA `weight` (so it scales the weighted sum *and* the normalize
   divisor, exactly as GTA intends) — **not** pre-multiplied into `delta_i`, and **no `√s`
   split**.
3. **Refactor** to `(up, down, alpha)` via `perform_lora_svd(merged, target_rank, dynamic_method,
   dynamic_param, …)`.
4. **Apply `lambda_` once** to the `up` factor (linear in the delta).

CLIP path is unchanged (it already uses a normalized weighted average + single magnitude
re-apply). Non-GTA UNet methods (slerp/karcher/nuslerp/sce/nearswap) keep their existing
mergekit path.

### Rank / refactor policy

- Default `target_rank = max(rank_i)` over the merged LoRAs.
- **Energy-dynamic rank wired in from the start**: expose the framework's dynamic selection
  (`dynamic_method="sv_cumulative"` with an energy threshold, capped at `target_rank`). Node
  surfaces a rank mode option: `max_rank` (fixed) or `energy` (dynamic, with threshold).
  **Default: `energy` at 0.99 retention** (capped at `max(rank_i)`), so magnitude lost to
  truncation is negligible while ranks stay bounded. The plan may tune this default.
- Alpha follows `perform_lora_svd`'s convention (`new_alpha = scale · new_rank`).
- Conv layers: `perform_lora_svd` already reshapes 4D → 2D and back; materialization must
  produce the matching 2D delta.

## Correctness properties this buys

1. **No squaring** — per-element normalize acts on the real delta once.
2. **Correct TIES sign vote** — elected on the delta, not the factors.
3. **Non-overlapping LoRAs keep full strength under normalize** — the per-element divisor
   counts only LoRAs active at that element (style+character that touch different features are
   not mutually diluted).
4. **Sign conflicts are arbitrated, not cancelled** — TIES elects a winner per element.
5. **`N` equal LoRAs scale ≈ 1/N, not 1/N²**.
6. **Correct per-LoRA `alpha`** — today only `alpha_0` (first LoRA) survives; materialization
   uses each LoRA's own alpha.
7. **Smaller, dependency-light merge core** — mergekit adapter ceremony removed for this family.

## Data flow (per UNet GTA key)

```
up_i, down_i, alpha_i   ──►  delta_i = (alpha_i/rank_i)·up_i@down_i     [materialize]
strengths s_i           ──►  gta_merge(deltas, weights=s_i, mode,…)     [own torch]
                                   │ sparsify per mode
                                   │ elect_sign (if consensus)
                                   │ disjoint_merge + per-element normalize
                                   │ rescale_norm (optional)
                                   ▼
                              merged delta  ──► perform_lora_svd(target_rank, energy)
                                                     │ (SVD, dynamic rank)
                                                     ▼
                                              up', down', alpha'  ──► ×lambda_ (on up')
                                                     ▼
                                              LoRAAdapter(up', down', alpha')
```

## Error handling / edge cases

- Zero merged delta (everything trimmed/cancelled) → `perform_lora_svd` already guards the
  zero-matrix case (rank 1); return a zero-ish adapter without crashing.
- `divisor == 0` per element (no surviving LoRA) → set to 1 (no division), matches mergekit.
- Rank clamp: `target_rank ≤ min(out, in)` and `≤ Σ rank_i` (merged delta rank bound).
- Breadcrumbs `gamma ≥ 1-density` edge case: clamp per the node's documented behavior.
- Mixed dtypes / device: materialize and SVD in float32 on the merge device, offload to CPU,
  matching the existing per-key offload pattern.
- Conv (4D) layers: reshape consistently for materialize + SVD.

## Performance

Materialize + SVD per key is heavier than the factored path (the factored design's original
motivation). Acceptable per the decision to prioritize correctness. Mitigations available in
plan: energy rank keeps output small; run in the existing thread pool; float32 on GPU when
available; optionally a randomized SVD path for very large layers (DiT) if full SVD proves slow.

## Testing

The repo's pytest suite does not collect in this environment (pre-existing import-path
breakage: `from validation import`, `from src.utility import`, `from types import`).
Verification is via **standalone numeric scripts**, plus new unit tests for `gta.py` that can
run in isolation.

Cases:
1. **Non-overlapping style+character** under normalize → each keeps ≈ full strength on its own
   elements (not `1/Σs` dilution).
2. **Sign conflict** → arbitrated to the weighted-majority winner, not cancelled to zero.
3. **`N` equal LoRAs** → contribution scales ≈ `1/N` (linear-normalize) and the delta magnitude
   does not collapse as `1/N²`.
4. **Refactor fidelity** → `up' @ down'` reconstructs `merged` within SVD truncation error at
   the chosen energy retention.
5. **Mode parity** → each sparsify variant (ties/dare/della/breadcrumbs) matches mergekit's
   corresponding method on full deltas within tolerance (dare/della are stochastic → compare
   statistics / fixed seed).
6. **Strength linearity** → scaling all `s_i` by `k` scales the merged delta by `k` (task
   arithmetic, normalize off) and preserves ratios (normalize on).

## Out of scope

- slerp, karcher, nuslerp, sce, nearswap (stay on mergekit).
- CLIP merge path (unchanged).
- Fixing the repo-wide pytest import breakage.

## Implementation deviations from design

- **`breadcrumbs` `default_rescale`**: mergekit's `BreadcrumbsMethod` has
  `default_rescale=False` (confirmed via `merge_methods/registry.py:69`), not `True` as
  listed in the placeholder. The implementation uses `getattr(method, "default_rescale", False)`
  so it inherits the correct value.
- **No separate memory doc** for `lora-merger-factored-strength-squaring` — that concept is
  already documented inline in this doc (§ Related prior work, § Decision summary) and the
  squaring-trap no longer applies to the GTA family since they now merge full deltas in
  `src/merge/gta.py` and refactor via SVD.
