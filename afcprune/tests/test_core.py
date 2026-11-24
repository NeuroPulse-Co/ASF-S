"""Architecture sizes (paper Table 4), surgery invariants, and the AFC subspace tools."""

import numpy as np
import pytest
import torch

from afcprune import afc, metrics, models, surgery
from test_golden import seeded_model


@pytest.mark.parametrize("name, n", [("conv2net", 183_434), ("conv6net", 815_018)])
def test_param_count(name, n):
    assert metrics.count_params(models.build(name)) == n


def test_flops_table4():
    assert metrics.count_flops(models.build("conv2net")) == 40_894_484   # paper 40.89M


def random_keep(m, frac, seed=0):
    rng = np.random.default_rng(seed)
    keep = {c: rng.random(getattr(m, c).out_channels) < frac for c, _, _ in m.CONV_SEQ}
    for v in keep.values():
        v[0] = True
    return keep


@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
@pytest.mark.parametrize("frac", [0.2, 0.6])
def test_prune_equals_masked(name, frac):
    m = seeded_model(name)
    keep = random_keep(m, frac)
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(surgery.prune(m, keep)(x), surgery.masked(m, keep)(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
def test_keep_all_is_identity(name):
    m = seeded_model(name)
    keep = {c: np.ones(getattr(m, c).out_channels, bool) for c, _, _ in m.CONV_SEQ}
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(surgery.prune(m, keep)(x), m(x))


def test_conv2net_table5_point_params():
    """(19, 16) kept reproduces Table 5's 75.8% params↓ (cifar10 step 7)."""
    m = models.build("conv2net")
    keep = {"conv1": np.arange(32) < 19, "conv2": np.arange(64) < 16}
    red = 100 * (1 - metrics.count_params(surgery.prune(m, keep)) / metrics.count_params(m))
    assert red == pytest.approx(75.84, abs=0.01)


# ── AFC subspace tools ────────────────────────────────────────

def test_principal_cos_invariances():
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.normal(size=(10, 3)))
    V = U[:, [2, 0, 1]] * np.array([-1, 1, -1])           # reorder + sign flips: same subspace
    assert afc.subspace_similarity(U, V) == pytest.approx(1.0)
    Q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    W = Q[:, :3]
    assert afc.subspace_similarity(U, W) < 0.9


def test_embedding_contains_pg1():
    rng = np.random.default_rng(1)
    _, W = afc.afc(rng.normal(size=(50, 10)) + 1, 0.8, 1e-6)
    g, _ = afc.pg1(W, 0.5)
    E, lam = afc.embedding(W, 0.5, 3)
    assert E.shape == (10, 3) and abs(abs(E[:, 0] @ g) - 1) < 1e-10
    assert np.all(np.diff(lam) <= 1e-12)


def two_module_acts(n=200, noise=0.1, seed=2):
    """Units 0–4 and 5–9 driven by two independent latent sources: a clear 2-module graph, large λ2−λ3."""
    rng = np.random.default_rng(seed)
    z = np.abs(rng.normal(size=(n, 2)))
    mix = np.zeros((2, 10)); mix[0, :5] = 1; mix[1, 5:] = 1
    return z @ mix + noise * np.abs(rng.normal(size=(n, 10)))


def test_bootstrap_pg1_stable_for_modular_data():
    b = afc.bootstrap_pg1(two_module_acts(), 0.5, 1e-6, 0.5, n_boot=20)
    assert b.shape == (20,) and b.min() > 0.95


def test_bootstrap_flags_unstable_pg1():
    """Unstructured units: no spectral gap, PG1 swaps between resamples — the P3 failure mode the
    bootstrap must expose (low |cos| for some draws)."""
    rng = np.random.default_rng(3)
    acts = np.abs(rng.normal(size=(50, 10)))
    b = afc.bootstrap_pg1(acts, 0.5, 1e-6, 0.5, n_boot=30)
    assert b.min() < 0.8
