"""
Golden regression (plan P13): afcprune reproduces the original cifar10/asfs code exactly on the frozen
inputs in golden/golden.npz (see golden/make_golden.py).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from afcprune import afc, contrast, models, surgery
from afcprune.scores import pgi

G = np.load(Path(__file__).parent / "golden" / "golden.npz")
SIGMA, EPS, ALPHA, K = 0.8, 1e-6, 0.5, 4


def seeded_model(name, seed=0):
    torch.manual_seed(seed)
    m = models.build(name)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.5, 0.5)
            mod.running_mean.uniform_(-0.2, 0.2); mod.running_var.uniform_(0.5, 2.0)
    return m.eval()


def test_afc_and_pg1():
    acts = G["acts"]
    for c in range(10):
        for s in range(2):
            _, W = afc.afc(acts[c, s], SIGMA, EPS)
            np.testing.assert_allclose(W, G["W"][c, s], atol=1e-12)
            g, lam = afc.pg1(W, ALPHA)
            np.testing.assert_allclose(g, G["g"][c, s], atol=1e-10)
            np.testing.assert_allclose(lam, G["lam"][c, s], atol=1e-12)


def test_median_sigma():
    acts = G["acts"]
    As = [afc.cosine_similarity(afc.activation_matrix(acts[c, s])) for c in range(10) for s in range(2)]
    assert afc.median_sigma(As) == pytest.approx(float(G["median_sigma"]), abs=1e-14)


def test_contrast_pca():
    dG, flip = contrast.contrast_matrix(G["g"][:, 0], G["g"][:, 1])
    np.testing.assert_allclose(dG, G["dG"], atol=1e-12)
    assert (flip == G["flip"]).all()
    comps, ratio, _ = contrast.pca(dG, K)
    np.testing.assert_allclose(comps, G["comps"], atol=1e-10)
    np.testing.assert_allclose(ratio, G["ratio"], atol=1e-12)


@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
@pytest.mark.parametrize("mode", ["aligned", "literal"])
def test_pgi(name, mode):
    m = seeded_model(name)
    for layer, v in pgi.pgi(m, G["comps"], mode).items():
        np.testing.assert_allclose(v, G[f"pgi_{mode}__{name}__{layer}"], atol=1e-10)


@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
def test_surgery(name):
    m = seeded_model(name)
    keep = {c: G[f"keep__{name}__{c}"] for c, _, _ in m.CONV_SEQ}
    with torch.no_grad():
        out = surgery.prune(m, keep)(torch.from_numpy(G["x"])).numpy()
    np.testing.assert_allclose(out, G[f"logits_pruned__{name}"], atol=1e-5)
