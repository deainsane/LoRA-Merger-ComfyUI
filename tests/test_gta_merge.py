# tests/test_gta_merge.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from gta_helpers import load_gta, run

gta = load_gta()


def test_elect_sign_weighted_majority():
    a = torch.tensor([2.0, -2.0])
    b = torch.tensor([-1.0, 1.0])
    w = torch.tensor([1.0, 1.0])
    wd = torch.stack([a * w[0], b * w[1]])
    sign = gta.elect_sign(wd)
    assert torch.equal(sign, torch.tensor([1.0, -1.0]))


def test_disjoint_normalize_nonoverlap_keeps_strength():
    a = torch.tensor([2.0, 0.0])
    b = torch.tensor([0.0, 2.0])
    w = torch.tensor([1.0, 1.0])
    merged = gta.disjoint_merge(torch.stack([a, b]), w, sign_consensus=True, normalize=True)
    assert torch.allclose(merged, torch.tensor([2.0, 2.0]))


def test_disjoint_normalize_overlap_conflict_arbitrated():
    a = torch.tensor([2.0])
    b = torch.tensor([-2.0])
    w = torch.tensor([1.0, 1.0])
    merged = gta.disjoint_merge(torch.stack([a, b]), w, sign_consensus=True, normalize=True)
    assert merged.abs().item() == 2.0


def test_linear_normalize_is_weighted_average():
    a = torch.tensor([1.0])
    b = torch.tensor([3.0])
    w = torch.tensor([1.0, 1.0])
    merged = gta.disjoint_merge(torch.stack([a, b]), w, sign_consensus=False, normalize=True)
    assert torch.allclose(merged, torch.tensor([2.0]))


def test_n_equal_loras_scale_one_over_n_not_squared():
    d = torch.tensor([1.0, -1.0, 0.5])
    stack = torch.stack([d, d, d, d])
    w = torch.ones(4)
    merged = gta.disjoint_merge(stack, w, sign_consensus=True, normalize=True)
    assert torch.allclose(merged, d)


# Runnable as a plain script (`python tests/<file>.py`); under pytest the
# test_* functions are collected directly, so the script runner must not fire
# at import time -- it calls sys.exit() and would abort collection.
if __name__ == "__main__":
    run([
        ("elect_sign", test_elect_sign_weighted_majority),
        ("nonoverlap_keeps_strength", test_disjoint_normalize_nonoverlap_keeps_strength),
        ("conflict_arbitrated", test_disjoint_normalize_overlap_conflict_arbitrated),
        ("linear_weighted_average", test_linear_normalize_is_weighted_average),
        ("n_loras_1_over_n", test_n_equal_loras_scale_one_over_n_not_squared),
    ])
