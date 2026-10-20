"""
Step 4 tests: Eq. 5–8 properties on synthetic affinity matrices.
"""

import numpy as np

from asfs.pg import canonical_sign, diffusion_operator, pg1_legacy, pg1_paper


def two_block_W(n1=4, n2=6, inside=0.9, across=0.05, eps=1e-6):
    n = n1 + n2
    W = np.full((n, n), across)
    W[:n1, :n1] = inside
    W[n1:, n1:] = inside
    return W + eps * np.eye(n)


def test_K_symmetric_and_alpha0_is_W():
    W = two_block_W()
    K = diffusion_operator(W, 0.5)
    np.testing.assert_allclose(K, K.T)
    np.testing.assert_allclose(diffusion_operator(W, 0.0), W)


def test_pg1_is_unit_eigenvector_of_K():
    rng = np.random.default_rng(0)
    B = rng.random((10, 10))
    W = (B + B.T) / 2 + 1e-6 * np.eye(10)
    g, lam = pg1_paper(W, 0.5)
    K = diffusion_operator(W, 0.5)
    np.testing.assert_allclose(np.linalg.norm(g), 1.0)
    np.testing.assert_allclose(K @ g, lam[1] * g, atol=1e-10)
    assert (np.diff(lam) <= 1e-12).all()                       # descending


def test_pg1_separates_two_blocks():
    g, _ = pg1_paper(two_block_W(), 0.5)
    s = np.sign(g)
    assert len(set(s[:4])) == 1 and len(set(s[4:])) == 1 and s[0] != s[4]


def test_pg1_invariant_to_unit_order():
    W = two_block_W()
    perm = np.random.default_rng(1).permutation(10)
    g, _ = pg1_paper(W, 0.5)
    gp, _ = pg1_paper(W[np.ix_(perm, perm)], 0.5)
    np.testing.assert_allclose(np.abs(gp), np.abs(g[perm]), atol=1e-10)


def test_canonical_sign():
    assert (canonical_sign(np.array([0.1, -0.9, 0.2])) == np.array([-0.1, 0.9, -0.2])).all()


def test_legacy_differs_from_paper_scaling():
    W = two_block_W()
    g_leg, lam_leg = pg1_legacy(W, 0.5)
    assert abs(np.linalg.norm(g_leg) - np.sqrt(lam_leg[1])) < 1e-8   # scaled by λ2^0.5, not unit
