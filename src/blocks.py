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
