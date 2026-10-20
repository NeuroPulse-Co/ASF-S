"""
Step 3 tests: Eq. 1–3 on hand-checkable inputs.
"""

import numpy as np

from asfs.afc import activation_matrix, afc, connectivity, cosine_similarity, gaussian_affinity


def test_activation_matrix_is_units_by_samples():
    acts = np.arange(12.0).reshape(4, 3)            # 4 samples, 3 units
    X = activation_matrix(acts)
    assert X.shape == (3, 4) and (X[1] == acts[:, 1]).all()


def test_cosine_known_values():
    X = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])   # 4 units × 2 samples
    A = cosine_similarity(X)
    np.testing.assert_allclose(np.diag(A), 1.0)
    assert A[0, 1] == 1.0 and A[0, 2] == 0.0 and A[0, 3] == -1.0
    np.testing.assert_allclose(A, A.T)


def test_cosine_scale_invariant():
    X = np.random.default_rng(0).normal(size=(5, 20))
    np.testing.assert_allclose(cosine_similarity(X * np.arange(1, 6)[:, None]), cosine_similarity(X))


def test_gaussian_affinity_values():
    sigma, eps = 0.1, 1e-6
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    W = gaussian_affinity(A, sigma, eps)
    np.testing.assert_allclose(np.diag(W), 1.0 + eps)
    np.testing.assert_allclose(W[0, 1], np.exp(-1.0 / (2 * sigma ** 2)))
    A2 = np.array([[1.0, 0.9], [0.9, 1.0]])
    np.testing.assert_allclose(gaussian_affinity(A2, sigma, 0)[0, 1], np.exp(-0.5))


def test_afc_shapes_and_symmetry():
    acts = np.random.default_rng(1).normal(size=(50, 10))
    A, W = afc(acts, 0.1, 1e-6)
    assert A.shape == W.shape == (10, 10)
    np.testing.assert_allclose(W, W.T)
    assert (W > 0).all() and W.max() <= 1 + 1e-6


def test_connectivity():
    assert connectivity(np.eye(4)) == 0.0
    assert connectivity(np.ones((4, 4))) == 1.0
