"""
Recovery tiers: closed-form ridge head, BN recalibration, LS compensation, and the gradient fine-tuning
modes. CPU, synthetic data, no dataset download.
"""

import copy

import numpy as np
import pytest
import torch

from afcprune import models, recovery
from afcprune.surgery import prune


def model_with_bn_stats(name, seed=0):
    torch.manual_seed(seed)
    m = models.build(name)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.5, 0.5)
            mod.running_mean.uniform_(-0.2, 0.2); mod.running_var.uniform_(0.5, 2.0)
    return m.eval()


def batches(n, bs=100, seed=0, labels=None):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, bs):
        x = torch.randn(min(bs, n - i), 3, 32, 32, generator=g)
        y = torch.randint(0, 10, (len(x),), generator=g)
        out.append((x, y))
    return out


def keep_frac(m, counts):
    return {c: np.arange(getattr(m, c).out_channels) < counts[c] for c, _, _ in m.CONV_SEQ}


def label_by(model, loader):
    """Relabel batches with the model's own predictions, so val accuracy is meaningful (100% for the model)."""
    with torch.no_grad():
        return [(x, model(x).argmax(1)) for x, _ in loader]


def all_bn_stats(m):
    return [(b.running_mean.clone(), b.running_var.clone()) for b in recovery._bns(m)]


# ── collect_features ──────────────────────────────────────────

@pytest.mark.parametrize("name, D", [("conv2net", 16384), ("conv6net", 256)])
def test_collect_features_shape(name, D):
    m = model_with_bn_stats(name)
    f, y = recovery.collect_features(m, batches(30, 10), max_batches=2)
    assert f.shape == (20, D) and f.dtype == torch.float64 and y.shape == (20,)
    # features reproduce the logits through the last FC layer
    fc = getattr(m, m.FC_LAYERS[-1])
    with torch.no_grad():
        ref = m(batches(30, 10)[0][0]).double()
    torch.testing.assert_close(f[:10] @ fc.weight.double().T + fc.bias.double(), ref, atol=1e-4, rtol=1e-4)


# ── T1: ridge head ────────────────────────────────────────────

@pytest.mark.parametrize("name, counts, n", [("conv6net", None, 1200), ("conv2net", {"conv1": 8, "conv2": 2}, 1200)])
def test_ridge_head_recovers_fc(name, counts, n):
    """Refit on the model's own logits: the fitted head must reproduce the original one."""
    dense = model_with_bn_stats(name)
    m = dense if counts is None else prune(dense, keep_frac(dense, counts))   # conv2net: D = 2·256 = 512
    teacher = copy.deepcopy(m)
    fc = getattr(m, m.FC_LAYERS[-1])
    W0, b0 = fc.weight.detach().clone(), fc.bias.detach().clone()
    torch.nn.init.normal_(fc.weight); torch.nn.init.zeros_(fc.bias)            # destroy the head
    train = batches(n, 200, seed=1)
    val = label_by(teacher, batches(300, 100, seed=2))
    res = recovery.ridge_head(m, train, val, [1e-10], target="logits", teacher=teacher)
    assert res["val_acc"] == 100.0 and res["lambda"] == 1e-10
    with torch.no_grad():
        x = val[0][0]
        torch.testing.assert_close(m(x), teacher(x), atol=1e-3, rtol=1e-3)
    # weights are identifiable only on feature columns that are active (dead ReLU units give zero columns)
    # and linearly independent; there they must equal the original head
    F_, _ = recovery.collect_features(teacher, train)
    active = F_.std(0) > 0
    Fa = F_[:, active] - F_[:, active].mean(0)
    if torch.linalg.matrix_rank(Fa) == int(active.sum()):
        torch.testing.assert_close(fc.weight[:, active], W0[:, active], atol=1e-3, rtol=1e-3)
    else:                                        # rank-deficient: equal on the span of the features
        torch.testing.assert_close((Fa @ (fc.weight[:, active] - W0[:, active]).double().T).abs().max().item(),
                                   0.0, atol=1e-2, rtol=0)


def test_ridge_lambda_selection_and_labels_target():
    m = model_with_bn_stats("conv6net")
    teacher = copy.deepcopy(m)
    lams = [1e-3, 1e-1, 10.0, 1e4]
    val = label_by(teacher, batches(300, 100, seed=2))
    train = label_by(teacher, batches(1000, 200, seed=1))
    res = recovery.ridge_head(m, train, val, lams, target="labels")
    assert res["lambda"] in lams and set(res["per_lambda"]) == set(lams)
    assert res["val_acc"] == max(res["per_lambda"].values())
    with pytest.raises(ValueError):
        recovery.ridge_head(m, train, val, lams, target="logits")              # no teacher


# ── T2: BN recalibration ──────────────────────────────────────

@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
def test_bn_recalibrate(name):
    m = model_with_bn_stats(name)
    before = all_bn_stats(m)
    loader = batches(200, 50)
    recovery.bn_recalibrate(m, loader, n_batches=4)
    after = all_bn_stats(m)
    assert all(not torch.allclose(a[0], b[0]) for a, b in zip(before, after))
    assert not m.training and all(b.momentum == 0.1 for b in recovery._bns(m))
    # stats already equal to the data's: recalibrating again on the same batches is idempotent
    recovery.bn_recalibrate(m, loader, n_batches=4)
    for (m1, v1), (m2, v2) in zip(after, all_bn_stats(m)):
        torch.testing.assert_close(m1, m2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v1, v2, atol=1e-5, rtol=1e-5)


# ── T3: LS compensation ───────────────────────────────────────

@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
def test_ls_compensate_keep_all_is_exact(name):
    dense = model_with_bn_stats(name)
    keep = {c: np.ones(getattr(dense, c).out_channels, bool) for c, _, _ in dense.CONV_SEQ}
    pruned = prune(dense, keep)
    errs = recovery.ls_compensate(dense, pruned, keep, batches(40, 10), n_batches=4, ridge=1e-8)
    assert set(errs) == {n for _, _, n in dense.CONV_SEQ if n is not None}
    for e in errs.values():
        assert e["before"] < 1e-8 and e["after"] < 1e-6


@pytest.mark.parametrize("name, frac", [("conv2net", 0.5), ("conv6net", 0.6)])
def test_ls_compensate_reduces_error(name, frac):
    dense = model_with_bn_stats(name)
    rng = np.random.default_rng(0)
    keep = {c: rng.random(getattr(dense, c).out_channels) < frac for c, _, _ in dense.CONV_SEQ}
    pruned = prune(dense, keep)
    errs = recovery.ls_compensate(dense, pruned, keep, batches(200, 50), n_batches=4)
    for e in errs.values():
        assert e["after"] < e["before"]
    # held-out check on the first compensated layer: its pre-BN output is closer to the dense one
    nxt = next(n for _, _, n in dense.CONV_SEQ if n is not None)
    ref = prune(dense, keep)
    x = batches(20, 20, seed=9)[0][0]
    outs = {}
    for tag, net in (("dense", dense), ("plain", ref), ("comp", pruned)):
        h = getattr(net, nxt).register_forward_hook(lambda _m, _i, o, tag=tag: outs.__setitem__(tag, o))
        with torch.no_grad():
            net(x)
        h.remove()
    tgt = outs["dense"][:, torch.as_tensor(keep[nxt])]
    assert (outs["comp"] - tgt).pow(2).sum() < (outs["plain"] - tgt).pow(2).sum()


# ── gradient fine-tuning modes ────────────────────────────────

def _params(m):
    return {k: v.detach().clone() for k, v in m.named_parameters()}


@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
def test_fc_frozen_changes_only_fc(name):
    m = model_with_bn_stats(name)
    p0, bn0 = _params(m), all_bn_stats(m)
    train, val = batches(64, 32, seed=1), batches(32, 32, seed=2)
    recovery.finetune(m, "fc_frozen", train, val, lr=1e-2, max_epochs=2, patience=5)
    p1 = _params(m)
    fc_names = tuple(f"{n}." for n in m.FC_LAYERS)
    changed = {k for k in p0 if not torch.equal(p0[k], p1[k])}
    assert changed and all(k.startswith(fc_names) for k in changed)
    for (a, b), (c, d) in zip(bn0, all_bn_stats(m)):
        assert torch.equal(a, c) and torch.equal(b, d)
    assert all(p.requires_grad for p in m.parameters()) and not m.training


def test_fc_mode_adapts_bn_stats():
    m = model_with_bn_stats("conv2net")
    p0, bn0 = _params(m), all_bn_stats(m)
    recovery.finetune(m, "fc", batches(64, 32, seed=1), batches(32, 32, seed=2), lr=1e-2, max_epochs=2, patience=5)
    p1 = _params(m)
    assert all(torch.equal(p0[k], p1[k]) for k in p0 if not k.startswith("fc."))
    assert any(not torch.equal(a, c) for (a, _), (c, _) in zip(bn0, all_bn_stats(m)))


def test_tiers_and_modes():
    assert set(recovery.TIERS) == {"T0", "T1", "T2", "T3", "T4"}
    with pytest.raises(ValueError):
        recovery.configure(models.build("conv2net"), "bogus")
