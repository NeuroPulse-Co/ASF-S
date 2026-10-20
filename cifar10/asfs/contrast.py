"""
contrast.py – class-wise PG1 contrast and its principal components [paper §4, Eq. 9–11].

  Δg_c = g_c^corr − g_c^incorr                    (Eq. 9)
  ΔG   = [Δg_1ᵀ; …; Δg_Cᵀ]  ∈ R^{C × N}           (Eq. 10)
  PCA(ΔG) → top-K components u_1 … u_K ∈ R^N      (Eq. 11)

Eigenvector signs are arbitrary, so g_corr and g_incorr can come out with opposite signs
for no reason; subtracting them then measures the sign flip, not a difference in
structure. The paper does not mention this. With align=True, g_incorr is flipped when
g_corr · g_incorr < 0 before Eq. 9 (the same fix as on cifar_100).

PCA is done with an SVD of the column-centred ΔG; this is what sklearn's PCA computes.
"""

import numpy as np


def contrast_matrix(g_corr, g_incorr, align=True):
    """g_corr, g_incorr: (C, N). Returns ΔG (C, N) and a (C,) bool array of flipped classes."""
    g_corr, g_incorr = np.asarray(g_corr, float), np.asarray(g_incorr, float)
    flip = np.sum(g_corr * g_incorr, 1) < 0 if align else np.zeros(len(g_corr), bool)
    return g_corr - np.where(flip[:, None], -g_incorr, g_incorr), flip


def pca(X, k):
    """
    X: (C, N). Returns (components (k, N) orthonormal rows, explained_variance_ratio (all),
    mean (N,)). Each component's sign is fixed so its largest-|·| entry is positive.
    """
    mean = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = s ** 2
    ratio = var / var.sum()
    comps = Vt[:k].copy()
    for i, v in enumerate(comps):
        if v[np.argmax(np.abs(v))] < 0:
            comps[i] = -v
    return comps, ratio, mean
