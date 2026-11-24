"""pipeline.contrast_from_acts reproduces the step-by-step afc → pg1 → contrast → pca chain."""

import numpy as np

from afcprune import afc, contrast, pipeline


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10), n // 10)
    acts = rng.normal(size=(n, 10)) + 2.0 * np.eye(10)[labels]
    correct = rng.random(n) < 0.7
    return acts, labels, correct


def test_matches_manual_chain():
    acts, labels, correct = synthetic()
    out = pipeline.contrast_from_acts(acts, labels, correct, S=50, sigma="median", seed=7)
    rng = np.random.default_rng(7)
    idx = [[pipeline.draw(labels, correct, c, s, 50, rng) for s in pipeline.SPLITS] for c in range(10)]
    A = [[afc.cosine_similarity(afc.activation_matrix(acts[i])) for i in row] for row in idx]
    sig = afc.median_sigma([a for row in A for a in row])
    assert out["sigma"] == sig
    g = np.array([[afc.pg1(afc.gaussian_affinity(A[c][s], sig, 1e-6), 0.5)[0] for s in range(2)]
                  for c in range(10)])
    np.testing.assert_allclose(out["g"], g)
    dG, _ = contrast.contrast_matrix(g[:, 0], g[:, 1])
    comps, _, _ = contrast.pca(dG, 4)
    np.testing.assert_allclose(out["comps"], comps)
    assert (out["counts"] == 50).all()


def test_bootstrap_diagnostic_shape():
    acts, labels, correct = synthetic()
    out = pipeline.contrast_from_acts(acts, labels, correct, S=50, n_boot=5)
    assert out["boot_median_abs_cos"].shape == (10, 2)
