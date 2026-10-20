"""
Step 7 tests: physical pruning computes the same function as the masked model, keeping
everything is the identity, and the shapes/parameter counts are what the arithmetic says.
"""

import numpy as np
import pytest
import torch

import models
from asfs.prune import keep_from_scores, keep_top_k, masked, prune


def model_with_bn_stats(name="conv2net", seed=0):
    torch.manual_seed(seed)
    m = models.build(name)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.5, 0.5)
            mod.running_mean.uniform_(-0.2, 0.2); mod.running_var.uniform_(0.5, 2.0)
    return m.eval()


def random_keep(m, frac, seed=0):
    rng = np.random.default_rng(seed)
    return {c: rng.random(getattr(m, c).out_channels) < frac for c, _, _ in m.CONV_SEQ}


@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
@pytest.mark.parametrize("frac", [0.2, 0.5, 0.9])
def test_prune_equals_mask(name, frac):
    m = model_with_bn_stats(name)
    keep = random_keep(m, frac)
    for k in keep.values():
        k[0] = True                                     # at least one per layer
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(prune(m, keep)(x), masked(m, keep)(x), atol=1e-4, rtol=1e-4)


def test_keep_all_is_identity():
    m = model_with_bn_stats()
    keep = {c: np.ones(getattr(m, c).out_channels, bool) for c, _, _ in m.CONV_SEQ}
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(prune(m, keep)(x), m(x), atol=0, rtol=0)


def test_pruned_shapes_and_params():
    m = model_with_bn_stats()
    keep = {"conv1": np.arange(32) < 10, "conv2": np.arange(64) < 20}
    p = prune(m, keep)
    assert p.conv1.out_channels == 10 and p.conv2.in_channels == 10 and p.conv2.out_channels == 20
    assert p.bn1.num_features == 10 and p.bn2.num_features == 20 and p.fc.in_features == 20 * 256
    expected = (3 * 9 * 10 + 10) + 2 * 10 + (10 * 9 * 20 + 20) + 2 * 20 + (20 * 256 * 10 + 10)
    assert sum(t.numel() for t in p.parameters()) == expected
    assert sum(t.numel() for t in m.parameters()) == 183_434          # original untouched


def test_keep_rules():
    s = np.array([0.0, 0.3, 0.29, 1.0])
    assert keep_from_scores(s, 0.3).tolist() == [False, True, False, True]   # Eq. 14: prune PGI < τ
    assert keep_from_scores(np.array([0.1, 0.2]), 0.9).tolist() == [False, True]   # never empty
    assert keep_top_k(s, 2).tolist() == [False, True, False, True]
