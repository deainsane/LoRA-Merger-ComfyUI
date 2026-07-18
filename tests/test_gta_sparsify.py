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


run([
    ("magnitude", test_magnitude_matches_mergekit),
    ("magnitude_rescale_l1", test_magnitude_rescale_l1_matches),
    ("outliers", test_outliers_matches_mergekit),
    ("bernoulli_seed", test_bernoulli_matches_with_seed),
    ("della_seed", test_della_matches_with_seed),
    ("density_one_identity", test_density_one_is_identity),
])