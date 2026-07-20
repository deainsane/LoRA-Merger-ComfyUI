# PM LoRA Merger: VRAM offload before merge

**Date:** 2026-07-20
**Status:** Approved, ready for planning
**Component:** `src/lora_mergekit_merge.py` (`LoraMergerMergekit`)

## Problem

On low-VRAM cards (e.g. 8 GB), the PM LoRA Merger node OOMs during its CUDA
merge compute because the workflow's DIT model (flux2, KREA2, etc.) — along
with CLIP and VAE — is already resident in VRAM. The merge needs GPU memory
for its own tensors (per-layer merge, rSVD refactor across an 8-worker pool),
and there isn't enough left after the base models are loaded.

## Key insight

The "reload after merge" half of the problem is free. ComfyUI's model manager
loads models to GPU lazily and on demand. Evicting them to CPU/offload before
the merge does **not** require us to reload them: the downstream KSampler
reloads whatever it needs when it executes after the merge. So the feature is
**unload-only**, and the merger node does **not** need the MODEL wired in as an
input — a global unload frees everything.

## Design

Add a single boolean widget to `LoraMergerMergekit` that evicts resident models
from VRAM at the start of the merge.

### 1. Widget

Added to `LoraMergerMergekit.INPUT_TYPES` `required`:

```python
"offload_models": ("BOOLEAN", {
    "default": True,
    "tooltip": (
        "Evict resident models (DIT/CLIP/VAE) from VRAM before merging, "
        "to avoid OOM on low-VRAM cards.\n"
        "ComfyUI reloads them automatically when the sampler runs afterward.\n"
        "Only acts when device=cuda; no effect on cpu merges."
    ),
})
```

Default `True`: best behavior out of the box for low-VRAM users, which is the
motivating case. High-VRAM users who don't want the unload/reload cycle toggle
it off.

### 2. Behavior

In `lora_mergekit()`, immediately after `device, dtype = map_device(device, dtype)`
and before any merge tensors are moved to CUDA:

```python
if offload_models and device.type == "cuda":
    logging.info("PM LoRA Merger: offloading resident models from VRAM before merge")
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
```

- `map_device` returns a `torch.device`, so `device.type == "cuda"` is the guard.
- When `device` is cpu, or `offload_models` is `False`: no-op.
- The existing post-merge `torch.cuda.empty_cache()` stays unchanged.
- `unload_all_models()` evicts across all torch devices globally; no MODEL
  reference required.

### 3. Signature

`lora_mergekit()` gains an `offload_models: bool = True` parameter. ComfyUI
passes widget values by name, so the new widget maps to this parameter.

## Trade-offs accepted

- On high-VRAM setups, default-on forces an unnecessary unload→reload cycle per
  merge. Mitigated by the widget toggle. Acceptable because the extension's
  motivating use is low-VRAM.
- Full unload (not selective) is used for simplicity and because on an 8 GB card
  freeing everything is what's actually needed. No per-model targeting.

## Testing

New tests in `tests/` (pytest, matching existing suite conventions):

1. `offload_models` is present in `LoraMergerMergekit.INPUT_TYPES()` with
   default `True`.
2. With `offload_models=True` and a cuda device, `unload_all_models` is called
   (spied/monkeypatched on `comfy.model_management`).
3. With `offload_models=False`, `unload_all_models` is **not** called.
4. With device `cpu` (and `offload_models=True`), `unload_all_models` is **not**
   called.

Tests monkeypatch `comfy.model_management.unload_all_models` /
`soft_empty_cache` so they run without a GPU and without actually touching model
state. The merge itself can be stubbed/minimal — these tests assert the offload
decision, not merge correctness.

## Out of scope

- Selective / per-model unloading or VRAM-threshold gating (YAGNI).
- A separate standalone "Free VRAM" utility node (considered; widget chosen).
- Any manual reload logic (handled by ComfyUI's lazy load).
