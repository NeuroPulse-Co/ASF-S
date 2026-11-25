"""
Step 6 tests: Eq. 12–13 scoring, the literal mode against the original code, and the
aligned mode's effective output vectors against a direct computation.
"""

import numpy as np
import torch

import models
from asfs.pgi import (bn_scale, effective_output_vectors, minmax, output_weight, pgi_aligned,
                      pgi_literal, scores)


def trained_like_model(seed=0):
    torch.manual_seed(seed)
    m = models.build("conv2net")
    for bn in (m.bn1, m.bn2):                                    # non-trivial BN statistics
        bn.weight.data.uniform_(0.5, 1.5)
        bn.running_var.uniform_(0.5, 2.0)
    return m.eval()


def comps(seed=1, k=4, c=10):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(c, k)))
    return q.T                                                    # (k, c) orthonormal rows


def test_scores_eq13_and_abs_of_mean():
    W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    T = np.array([[1.0, 1.0], [1.0, -1.0]])                       # s = [[1,1],[1,-1],[0,2]]
    np.testing.assert_allclose(scores(W, T), minmax([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(scores(W, T, abs_of_mean=True), minmax([1.0, 0.0, 1.0]))


def test_minmax():
    np.testing.assert_allclose(minmax([2.0, 4.0, 3.0]), [0.0, 1.0, 0.5], atol=1e-9)


def original_compute_importance(conv_flat, fc_w, components):
    """Verbatim from min_scripts/pg_pruning_conv2.py:94-108."""
    avg_fc = np.mean(fc_w, axis=0)
    scores_ = []
    for comp in components:
        proj_size = min(conv_flat.shape[1], avg_fc.shape[0], comp.shape[0])
        conv_proj = conv_flat[:, :proj_size]
        fc_proj = comp[:proj_size] * avg_fc[:proj_size]
        scores_.append(conv_proj @ fc_proj)
    scores_ = np.array(scores_)
    importance = np.abs(scores_.mean(axis=0))
    return (importance - importance.min()) / (importance.max() - importance.min() + 1e-12)


def test_literal_matches_original_code():
    m, u = trained_like_model(), comps()
    ours = pgi_literal(m, u, abs_of_mean=True)
    fc_w = m.fc.weight.detach().double().numpy()
    for conv in ("conv1", "conv2"):
        flat = getattr(m, conv).weight.detach().double().numpy().reshape(getattr(m, conv).out_channels, -1)
        np.testing.assert_allclose(ours[conv], original_compute_importance(flat, fc_w, u), atol=1e-9)


def test_effective_vectors_last_conv():
    m = trained_like_model()
    E = effective_output_vectors(m)
    fc = m.fc.weight.detach().double().numpy()                   # (10, 64*256)
    a2 = bn_scale(m.bn2)
    i = 7
    np.testing.assert_allclose(E["conv2"][i], a2[i] * fc[:, i * 256:(i + 1) * 256].mean(1), atol=1e-12)


def test_effective_vectors_first_conv():
    m = trained_like_model()
    E = effective_output_vectors(m)
    W2 = m.conv2.weight.detach().double().numpy()
    a1 = bn_scale(m.bn1)
    j = 3
    ref = a1[j] * sum(W2[i, j].sum() * E["conv2"][i] for i in range(64))
    np.testing.assert_allclose(E["conv1"][j], ref, atol=1e-10)
    assert E["conv1"].shape == (32, 10) and E["conv2"].shape == (64, 10)


def test_aligned_uses_column_mean_of_output_weight():
    m, u = trained_like_model(), comps()
    E = effective_output_vectors(m)
    w_bar = output_weight(m).mean(1)
    ref = scores(E["conv2"], u * w_bar[None, :])
    np.testing.assert_allclose(pgi_aligned(m, u)["conv2"], ref)
    assert set(pgi_aligned(m, u)) == {"conv1", "conv2"}
