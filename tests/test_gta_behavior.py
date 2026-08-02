import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
import torch
from gta_helpers import load_gta, gta, run

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


def _delta_from_lora(up, down, alpha):
    rank = up.shape[1]
    return (alpha / rank) * (up @ down)


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


def test_style_plus_character_nonoverlap_keeps_strength():
    torch.manual_seed(0)
    A = torch.zeros(4, 6); A[0, :] = 2.0
    B = torch.zeros(4, 6); B[2, :] = 2.0
    merged = gta.gta_merge([A, B], torch.tensor([1.0, 1.0]), mode="ties",
                           density=1.0, normalize=True)
    assert torch.allclose(merged[0], A[0])
    assert torch.allclose(merged[2], B[2])


def test_normalize_does_not_collapse_as_1_over_n_squared():
    torch.manual_seed(0)
    deltas = [torch.randn(6, 6) for _ in range(4)]
    avg = torch.stack(deltas).mean(0)  # snapshot: gta_merge consumes `deltas`
    merged = gta.gta_merge(deltas, torch.ones(4), mode="ties", density=1.0,
                           normalize=True)
    assert merged.norm() > 0.5 * avg.norm()


# Runnable as a plain script (`python tests/<file>.py`); under pytest the
# test_* functions are collected directly, so the script runner must not fire
# at import time -- it calls sys.exit() and would abort collection.
if __name__ == "__main__":
    run([
        ("refactor_reconstructs", test_refactor_reconstructs_delta),
        ("nonoverlap_keeps_strength", test_style_plus_character_nonoverlap_keeps_strength),
        ("no_1_over_n_squared", test_normalize_does_not_collapse_as_1_over_n_squared),
    ])
