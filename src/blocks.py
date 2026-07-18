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
