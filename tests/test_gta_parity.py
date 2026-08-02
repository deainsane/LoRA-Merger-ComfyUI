import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from gta_helpers import load_gta, run

from mergekit.common import ModelReference, ModelPath, ImmutableMap
from mergekit.architecture import WeightInfo
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods import REGISTERED_MERGE_METHODS
from mergekit.merge_methods.generalized_task_arithmetic import GTATask
from mergekit.sparsify import RescaleNorm

gta = load_gta()


def _mk_gta(deltas, strengths, mode, normalize, density=1.0, rescale=None):
    """Run mergekit's GTATask on full deltas with a zeros base (LoRA convention)."""
    tensors = {}
    params = {}
    base_ref = ModelReference(model=ModelPath(path="zeros.base"))
    tensors[base_ref] = torch.zeros_like(deltas[0])
    params[base_ref] = ImmutableMap({"weight": 0.0, "density": density})
    for i, (d, s) in enumerate(zip(deltas, strengths)):
        r = ModelReference(model=ModelPath(path=f"m.{i}"))
        tensors[r] = d
        params[r] = ImmutableMap({"weight": s, "density": density})
    wi = WeightInfo(name="w", dtype=None, is_embed=False)
    gt = GatherTensors(weight_info=ImmutableMap({r: WeightInfo(name=f"m{i}.w")
                                                 for i, r in enumerate(tensors)}))
    method = REGISTERED_MERGE_METHODS[mode]
    task = GTATask(method=method, tensors=gt, base_model=base_ref, weight_info=wi,
                   gather_tensors=gt, tensor_parameters=ImmutableMap(params),
                   int8_mask=False, normalize=normalize, lambda_=1.0,
                   rescale_norm=rescale)
    return task.execute(tensors=tensors)


def _check(mode, mk_mode, normalize, density=1.0):
    torch.manual_seed(0)
    deltas = [torch.randn(6, 10), torch.randn(6, 10), torch.randn(6, 10)]
    strengths = [1.0, 0.7, 0.4]
    exp = _mk_gta(deltas, strengths, mk_mode, normalize, density)
    got = gta.gta_merge(deltas, torch.tensor(strengths), mode=mode,
                        normalize=normalize, density=density)
    assert torch.allclose(got, exp, atol=1e-5), (
        f"{mode}/{mk_mode} normalize={normalize} density={density} "
        f"maxdiff={ (got-exp).abs().max() }")


def test_task_arithmetic_parity():
    _check("task_arithmetic", "task_arithmetic", normalize=True)
    _check("task_arithmetic", "task_arithmetic", normalize=False)


def test_ties_parity():
    _check("ties", "ties", normalize=True, density=0.6)
    _check("ties", "ties", normalize=False, density=0.6)


def test_resolve_rescale_default_matches_mergekit():
    assert gta.resolve_rescale_norm("della", "default") in (None, "l1")
    assert gta.resolve_rescale_norm("ties", "none") is None
    assert gta.resolve_rescale_norm("ties", "l2") == "l2"


# Runnable as a plain script (`python tests/<file>.py`); under pytest the
# test_* functions are collected directly, so the script runner must not fire
# at import time -- it calls sys.exit() and would abort collection.
if __name__ == "__main__":
    run([
        ("task_arithmetic_parity", test_task_arithmetic_parity),
        ("ties_parity", test_ties_parity),
        ("resolve_rescale_default_matches_mergekit", test_resolve_rescale_default_matches_mergekit),
    ])
