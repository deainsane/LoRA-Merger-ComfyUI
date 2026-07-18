import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
import torch
from gta_helpers import load_gta, run

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def _load(modfile, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "src", modfile))
    m = importlib.util.module_from_spec(spec)
    m.__name__ = name
    m.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_refactor_reconstructs_delta():
    torch.manual_seed(0)
    U = torch.randn(32, 8)
    V = torch.randn(8, 48)
    S_diag = torch.tensor([10.0, 8.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1])
    delta = U @ torch.diag(S_diag) @ V
    refactor = _load("merge/lora_refactor.py", "merge.lora_refactor")
    up, down, alpha = refactor.merged_delta_to_lora(delta, target_rank=16,
                                                    energy=0.999)
    rank = up.shape[1]
    recon = (alpha / rank) * (up @ down)
    err = (recon - delta).norm() / delta.norm()
    assert err < 0.05, f"reconstruction error {err}"


run([("refactor_reconstructs", test_refactor_reconstructs_delta)])