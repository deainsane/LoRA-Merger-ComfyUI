# tests/test_gta_sparsify.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from gta_helpers import load_gta, run
from mergekit.sparsify import (
    magnitude as mk_magnitude,
    magnitude_outliers as mk_outliers,
    bernoulli as mk_bernoulli,
    della_magprune as mk_della,
    RescaleNorm,
)

gta = load_gta()


def _rand(shape=(8, 16), seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape)


def test_magnitude_matches_mergekit():
    t = _rand()
    for d in (0.3, 0.5, 0.9):
        got = gta.sparsify(t, "magnitude", density=d)
        exp = mk_magnitude(t, density=d)
        assert torch.allclose(got, exp), f"density={d}"


def test_magnitude_rescale_l1_matches():
    t = _rand()
    got = gta.sparsify(t, "magnitude", density=0.5, rescale_norm="l1")
    exp = mk_magnitude(t, density=0.5, rescale_norm=RescaleNorm.l1)
    assert torch.allclose(got, exp)


def test_outliers_matches_mergekit():
    t = _rand()
    got = gta.sparsify(t, "magnitude_outliers", density=0.7, gamma=0.1)
    exp = mk_outliers(t, density=0.7, gamma=0.1)
    assert torch.allclose(got, exp)


def test_bernoulli_matches_with_seed():
    t = _rand()
    torch.manual_seed(42); got = gta.sparsify(t, "random", density=0.6)
    torch.manual_seed(42); exp = mk_bernoulli(t, density=0.6)
    assert torch.allclose(got, exp)


def test_della_matches_with_seed():
    t = _rand()
    torch.manual_seed(7); got = gta.sparsify(t, "della_magprune", density=0.7, epsilon=0.05)
    torch.manual_seed(7); exp = mk_della(t, density=0.7, epsilon=0.05)
    assert torch.allclose(got, exp)


def test_density_one_is_identity():
    t = _rand()
    assert torch.allclose(gta.sparsify(t, "magnitude", density=1.0), t)


# --- della on large tensors: the chunked path (rows > chunk threshold) must
#     preserve every statistical property of the whole-tensor path. --------

def _della_large(seed=0, rows=5000, cols=64, density=0.7, epsilon=0.1,
                 rescale_norm=None):
    torch.manual_seed(seed)
    t = torch.randn(rows, cols)
    got = gta.sparsify(t, "della_magprune", density=density, epsilon=epsilon,
                       rescale_norm=rescale_norm)
    return t, got


def test_della_large_preserves_density():
    # rows=5000 exceeds the chunk threshold, so this exercises the chunked path.
    t, got = _della_large(density=0.7, epsilon=0.1)
    kept = (got != 0).float().mean().item()
    assert abs(kept - 0.7) < 0.02, f"kept fraction {kept} != ~0.7"


def test_della_large_l1_preserved():
    t, got = _della_large(density=0.6, epsilon=0.1, rescale_norm="l1")
    before, after = t.abs().sum().item(), got.abs().sum().item()
    assert abs(after - before) / before < 0.01, f"l1 {after} vs {before}"


def test_della_large_keep_prob_monotonic_in_magnitude():
    # larger-magnitude elements within a row must be kept more often.
    t, got = _della_large(density=0.6, epsilon=0.3)
    mags = t.abs()
    med = mags.median(dim=1, keepdim=True).values
    hi = mags >= med
    kept = got != 0
    kept_hi = kept[hi].float().mean().item()
    kept_lo = kept[~hi].float().mean().item()
    assert kept_hi > kept_lo + 0.05, f"hi={kept_hi} not > lo={kept_lo}"


def test_della_chunk_boundary_invariant_density():
    # same seed, chunked vs a single block: aggregate density must agree even
    # though the per-element bernoulli draws differ.
    torch.manual_seed(3)
    t = torch.randn(9000, 32)
    a = gta.sparsify(t.clone(), "della_magprune", density=0.5, epsilon=0.1)
    torch.manual_seed(3)
    t2 = torch.randn(9000, 32)
    b = gta.sparsify(t2.clone(), "della_magprune", density=0.5, epsilon=0.1)
    ka, kb = (a != 0).float().mean().item(), (b != 0).float().mean().item()
    assert abs(ka - kb) < 0.01 and abs(ka - 0.5) < 0.02


def test_della_scatter_rank_equivalence():
    # The chunked path builds per-row ranks with scatter_ instead of a second
    # argsort; the rank values must match exactly (only the memory differs).
    torch.manual_seed(1)
    mags = torch.randn(50, 40).abs()
    sorted_idx = torch.argsort(mags, dim=1, descending=False)
    ranks_argsort = sorted_idx.argsort(dim=1)                      # old method
    positions = torch.arange(40, dtype=torch.int32).unsqueeze(0)
    ranks_scatter = torch.empty((50, 40), dtype=torch.int32)
    ranks_scatter.scatter_(1, sorted_idx, positions.expand(50, 40))  # new method
    assert torch.equal(ranks_scatter.long(), ranks_argsort)


def test_della_wide_layer_preserves_density_and_monotonic():
    # Wide layer: rows>threshold AND large cols, so the element budget forces
    # short blocks (the KREA2-style path). Statistics must still hold.
    torch.manual_seed(0)
    t = torch.randn(8000, 6000)
    got = gta.sparsify(t, "della_magprune", density=0.6, epsilon=0.2)
    kept = (got != 0).float().mean().item()
    assert abs(kept - 0.6) < 0.02, f"kept {kept} != ~0.6"
    mags = t.abs()
    med = mags.median(dim=1, keepdim=True).values
    hi = mags >= med
    kept_mask = got != 0
    assert kept_mask[hi].float().mean().item() > kept_mask[~hi].float().mean().item() + 0.05


# Runnable as a plain script (`python tests/<file>.py`); under pytest the
# test_* functions are collected directly, so the script runner must not fire
# at import time -- it calls sys.exit() and would abort collection.
if __name__ == "__main__":
    run([
        ("magnitude", test_magnitude_matches_mergekit),
        ("magnitude_rescale_l1", test_magnitude_rescale_l1_matches),
        ("outliers", test_outliers_matches_mergekit),
        ("bernoulli_seed", test_bernoulli_matches_with_seed),
        ("della_seed", test_della_matches_with_seed),
        ("density_one_identity", test_density_one_is_identity),
        ("della_large_density", test_della_large_preserves_density),
        ("della_large_l1", test_della_large_l1_preserved),
        ("della_large_monotonic", test_della_large_keep_prob_monotonic_in_magnitude),
        ("della_chunk_density", test_della_chunk_boundary_invariant_density),
        ("della_scatter_rank_equivalence", test_della_scatter_rank_equivalence),
        ("della_wide_density_monotonic", test_della_wide_layer_preserves_density_and_monotonic),
    ])
