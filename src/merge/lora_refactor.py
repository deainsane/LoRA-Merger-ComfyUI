"""Refactor a merged full delta back into a LoRA (up, down, alpha) via SVD.

Wraps the existing SVD framework (utility.perform_lora_svd) with energy-based
dynamic rank selection. Imports `utility` (only usable inside the ComfyUI
runtime / with src on sys.path), so it is kept separate from the dependency-free
gta.py."""
from typing import Optional, Tuple
import importlib.util
import os
import sys
import types

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")


def _setup_src_pkg():
    """Stub src/ and merge/ packages so utility.py's relative imports resolve."""
    class LoRAAdapterStub:
        def __init__(self, *args, **kwargs):
            self.state_dict = {}

    class LoRAAdapterModuleStub:
        LoRAAdapter = LoRAAdapterStub

    comfy_stub = types.ModuleType("comfy")
    comfy_weight_adapter_stub = types.ModuleType("comfy.weight_adapter")
    comfy_weight_adapter_stub.LoRAAdapter = LoRAAdapterStub
    comfy_stub.weight_adapter = comfy_weight_adapter_stub
    sys.modules.setdefault("comfy", comfy_stub)
    sys.modules.setdefault("comfy.weight_adapter", comfy_weight_adapter_stub)

    stub_types = types.ModuleType("types")
    from typing import Dict, Tuple, Union
    stub_types.LORA_TENSORS = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    stub_types.DeviceType = Union[torch.device, str]
    stub_types.DtypeType = Union[torch.dtype, str]
    stub_types.MIN_SINGULAR_VALUE = 1e-6
    stub_mergedelta = types.ModuleType("merge.delta")
    stub_merge = types.ModuleType("merge")
    stub_merge.delta = stub_mergedelta
    stub_mergedelta.__package__ = "merge"
    stub_merge.__path__ = [os.path.join(_SRC, "merge")]
    stub_merge.__package__ = "merge"
    stub_src = types.ModuleType("src")
    stub_src.__path__ = [_SRC]
    stub_src.__package__ = "src"
    for n, m in [("types", stub_types), ("src", stub_src), ("merge", stub_merge),
                 ("merge.delta", stub_mergedelta)]:
        sys.modules.setdefault(n, m)


def _load_perform_lora_svd():
    _setup_src_pkg()
    util_path = os.path.join(_SRC, "utility.py")
    spec = importlib.util.spec_from_file_location("src.utility", util_path)
    util = importlib.util.module_from_spec(spec)
    util.__name__ = "src.utility"
    util.__package__ = "src"
    sys.modules["src.utility"] = util
    spec.loader.exec_module(util)
    return util.perform_lora_svd


try:
    from ..utility import perform_lora_svd
except ImportError:
    perform_lora_svd = _load_perform_lora_svd()


def merged_delta_to_lora(delta: torch.Tensor, target_rank: int, *,
                         energy: Optional[float] = 0.99, device: str = "cpu",
                         dtype: torch.dtype = torch.float32
                         ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Decompose `delta` into (up, down, alpha).

    target_rank: hard cap on output rank (e.g. max input rank).
    energy: cumulative singular-value retention for dynamic rank (None = fixed rank).
    """
    if energy is None:
        up, down, alpha = perform_lora_svd(delta, target_rank=target_rank,
                                           device=device, dtype=dtype, scale=1.0)
    else:
        up, down, alpha = perform_lora_svd(delta, target_rank=target_rank,
                                           device=device, dtype=dtype, scale=1.0,
                                           dynamic_method="sv_cumulative",
                                           dynamic_param=energy)
    return up, down, alpha