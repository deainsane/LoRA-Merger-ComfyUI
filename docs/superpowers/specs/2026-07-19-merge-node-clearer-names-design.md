# Clearer widget names for the merge nodes

Date: 2026-07-19
Status: Approved (design)

## Goal

Make the merge-workflow node parameters self-explanatory by renaming three
cryptic user-facing widget keys. **Naming only** — no behavior change, no
reordering, no hiding of widgets, no default changes. The polarity/semantics of
every renamed control stay identical.

Out of scope (explicitly deferred): collapsing "advanced" knobs, moving knobs
between nodes, unifying the five strength controls, and any default changes.

## The renames

| node(s) | old widget key | new widget key | semantics |
| --- | --- | --- | --- |
| DARE, Breadcrumbs, DELLA | `ties` | `sign_consensus` | unchanged — ON elects a per-element sign and keeps only agreeing contributions |
| Linear, Task Arithmetic, TIES, DARE, Breadcrumbs, DELLA | `normalize` | `average_weights` | unchanged — ON divides by the summed weights (weighted average); OFF is an additive sum |
| PM LoRA Merger | `lambda_` | `output_scale` | unchanged — global scale applied to the final merged result |

## Why this is safe for saved workflows

ComfyUI serializes each node's widget values as a **positional list**
(`widgets_values`), mapped to the current `INPUT_TYPES` widgets **by order, not
by name** (verified against saved workflow JSON in this ComfyUI install). So a
rename that keeps the widget in the **same position** preserves the stored value
in every saved *graph* workflow — only the visible label changes.

Known caveat: an *API-format* prompt (name-keyed `inputs` JSON) that hardcodes
`"ties"` / `"normalize"` / `"lambda_"` will fall back to the widget default,
because ComfyUI filters out inputs not declared in `INPUT_TYPES` (so a
kwargs-alias shim in the node function cannot receive them without re-declaring
the old widget). Accepted for the sake of clean names; interactive graph users
are unaffected.

## Mechanics

The renames touch only the user-facing edge. **Internal plumbing keys are left
unchanged**, so the merge pipeline needs no downstream edits.

### Boolean toggles (`nodes_merge_methods.py`)

For each affected method node:

1. `INPUT_TYPES`: rename the key string in place (same position, same
   `("BOOLEAN", {..., "tooltip": ...})` spec — tooltips already improved in a
   prior commit).
2. `get_method(...)`: rename the parameter (`ties` -> `sign_consensus`,
   `normalize` -> `average_weights`).
3. The emitted `settings` dict is **unchanged**: it still writes
   `"sign_consensus_algorithm": sign_consensus` and `"normalize": average_weights`.
   Therefore `dispatcher.py`, `process_key`, `gta_merge`, the sweep sampler and
   the validators keep reading `method_args['normalize']` and
   `'sign_consensus_algorithm'` exactly as today.

Note: the `ties` -> `sign_consensus` rename applies only to the three nodes that
expose the optional toggle (DARE, Breadcrumbs, DELLA). The TIES node has no such
widget (sign consensus is always on); Linear and Task Arithmetic never had one.

### `output_scale` (`lora_mergekit_merge.py`)

`lambda_` is used as an internal key across many files (`dispatcher.py`,
`algorithms.py`, `types.py` `MergeContext`, `lora_parameter_sweep_sampler.py`,
`validation/validators.py`, the `merge_context` dict). Only the **widget** and
the node's **entry function parameter** are renamed; the internal key stays
`lambda_`:

1. `INPUT_TYPES`: `lambda_` -> `output_scale` (same position — it is the first
   widget on the Merger node).
2. Entry function `lora_mergekit(...)`: parameter `lambda_` -> `output_scale`.
3. At the internal boundary, alias once: `self.merge(..., lambda_=output_scale, ...)`.
   Everything downstream (the `merge()` signature, `merge_context["lambda_"]`,
   `algorithms.py` `method_args.get('lambda_')`, sweep sampler `context['lambda_']`,
   validators) is untouched.

The experimental `src/experimental/checkpoint_merge.py` has its own separate
`lambda_` widget on a different node and is **out of scope**.

## Testing

A standalone script test (repo pytest is broken; follow the existing
`tests/test_gta_*.py` pattern) asserting, for each affected node:

- `INPUT_TYPES` exposes the new key (`sign_consensus` / `average_weights` /
  `output_scale`) and no longer exposes the old key, at the **same index** among
  its widgets as the old key occupied.
- `get_method(...)` / `lora_mergekit(...)` called with the new parameter name
  produces the unchanged internal keys: `sign_consensus_algorithm`,
  `normalize`, and (for the Merger) an unchanged `lambda_` in the merge context.

No changes to the existing `gta.py` behavior tests are expected, since internal
keys and math are unchanged.

## Rollout note

Because saved graphs restore widget values positionally, existing user workflows
keep working with no migration step. The visible labels update on next load.
