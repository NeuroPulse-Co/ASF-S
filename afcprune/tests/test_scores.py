"""Baseline scorers: shapes, determinism, non-mutation, and per-criterion sanity. CPU, synthetic data."""

import copy

import numpy as np
import pytest
import torch

from afcprune import models
from afcprune.scores import ScoreContext, get, names
from afcprune.surgery import masked

WEIGHT = ["l1", "l2", "random", "bn_gamma", "fpgm"]
DATA = ["taylor", "hrank", "act_mean"]


def make_model(name="conv2net", seed=0):
    torch.manual_seed(seed)
    m = models.build(name)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.2, 0.5)
            mod.running_mean.uniform_(-0.2, 0.2); mod.running_var.uniform_(0.5, 2.0)
    return m.eval()


def calib(n_batches=2, bs=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(bs, 3, 32, 32, generator=g), torch.randint(0, 10, (bs,), generator=g))
            for _ in range(n_batches)]


def ctx(seed=0):
    return ScoreContext(device="cpu", calib=calib(), seed=seed)


def test_registered():
    assert set(WEIGHT + DATA) <= set(names())


@pytest.mark.parametrize("arch", ["conv2net", "conv6net"])
@pytest.mark.parametrize("name", WEIGHT + DATA)
def test_shapes_finite_deterministic_no_mutation(arch, name):
    m = make_model(arch)
    before = copy.deepcopy(m.state_dict())
    s1, s2 = get(name)(m, ctx()), get(name)(m, ctx())
    assert list(s1) == [c for c, _, _ in m.CONV_SEQ]
    for c, v in s1.items():
        assert v.shape == (getattr(m, c).out_channels,)
        assert np.isfinite(v).all()
        np.testing.assert_array_equal(v, s2[c])
    after = m.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not m.training


def test_l1_l2_manual():
    m = make_model()
    w = m.conv2.weight.detach().double()
    np.testing.assert_allclose(get("l1")(m, ctx())["conv2"], w.abs().sum((1, 2, 3)).numpy())
    np.testing.assert_allclose(get("l2")(m, ctx())["conv2"], w.pow(2).sum((1, 2, 3)).sqrt().numpy())


def test_random_seeded():
    m = make_model()
    a, b = get("random")(m, ctx(0))["conv1"], get("random")(m, ctx(1))["conv1"]
    assert not np.allclose(a, b)


def test_bn_gamma():
    m = make_model()
    np.testing.assert_allclose(get("bn_gamma")(m, ctx())["conv1"], m.bn1.weight.detach().abs().double().numpy())


def test_fpgm_duplicate_filter_is_least_important():
    m = make_model()
    with torch.no_grad():
        mean = m.conv1.weight.mean(0)
        m.conv1.weight[3] = mean            # the filter closest to all others
    s = get("fpgm")(m, ctx())["conv1"]
    assert s.argmin() == 3


@pytest.mark.parametrize("layer, ch", [("conv1", 5), ("conv2", 7)])
def test_taylor_zero_for_disconnected_channel(layer, ch):
    m = make_model()
    keep = {c: np.ones(getattr(m, c).out_channels, bool) for c, _, _ in m.CONV_SEQ}
    keep[layer][ch] = False
    s = get("taylor")(masked(m, keep), ctx())[layer]
    assert s[ch] == 0.0
    assert (np.delete(s, ch) > 0).any()


def test_hrank_bounds():
    m = make_model()
    s = get("hrank")(m, ctx())
    for v in s.values():
        assert (v >= 0).all() and (v <= 32).all()


def test_needs_calib():
    with pytest.raises(ValueError):
        get("taylor")(make_model(), ScoreContext())
