"""
Step 5 tests: Eq. 9–11 on hand-checkable inputs, and agreement with sklearn's PCA.
"""

import numpy as np
from sklearn.decomposition import PCA

from asfs.contrast import contrast_matrix, pca


def test_contrast_is_difference():
    gc = np.array([[1.0, 0.0], [0.6, 0.8]])
    gi = np.array([[0.8, 0.6], [0.0, 1.0]])
    dG, flip = contrast_matrix(gc, gi)
    np.testing.assert_allclose(dG, gc - gi)
    assert not flip.any()


def test_contrast_aligns_sign():
    gc = np.array([[1.0, 0.0]])
    gi = np.array([[-0.8, -0.6]])                    # same direction as (0.8, 0.6), flipped sign
    dG, flip = contrast_matrix(gc, gi, align=True)
    assert flip.tolist() == [True]
    np.testing.assert_allclose(dG, [[0.2, -0.6]])
    dG_raw, _ = contrast_matrix(gc, gi, align=False)
    np.testing.assert_allclose(dG_raw, [[1.8, 0.6]])


def test_pca_matches_sklearn():
    X = np.random.default_rng(0).normal(size=(10, 10))
    comps, ratio, mean = pca(X, 4)
    sk = PCA(n_components=10).fit(X)
    np.testing.assert_allclose(ratio, sk.explained_variance_ratio_, atol=1e-10)
    np.testing.assert_allclose(np.abs(comps), np.abs(sk.components_[:4]), atol=1e-8)
    np.testing.assert_allclose(comps @ comps.T, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(mean, X.mean(0))


def test_pca_rank_one():
    u = np.array([3.0, 4.0]) / 5
    X = np.outer(np.arange(5.0), u)                  # all rows along u
    comps, ratio, _ = pca(X, 1)
    np.testing.assert_allclose(ratio[0], 1.0)
    np.testing.assert_allclose(comps[0], u, atol=1e-12)
