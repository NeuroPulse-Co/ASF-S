"""
pg.py – first principal gradient PG1 by diffusion-map embedding [paper §4, Eq. 5–8].

  D = diag(W 1)                    (Eq. 5)
  K = D^-α W D^-α,  α ∈ [0, 1]     (Eq. 6)  – symmetric, because W is
  K v_k = λ_k v_k,  λ_1 ≥ λ_2 ≥ …  (Eq. 7)
  g = v_2, unit length             (Eq. 8)

Modes:
  paper   eigh on the symmetric K, exactly Eq. 5–8. The sign of an eigenvector is
          arbitrary; it is fixed so that the largest-|·| entry is positive. Step 5 still
          aligns correct/incorrect pairs before subtracting them.
  legacy  what the original pg_ext_2.py computes: K row-normalised to a Markov matrix M,
          numpy eigh on that non-symmetric M (eigh reads only its lower triangle), second
          largest eigenvector scaled by λ_2^t with t = 0.5. Kept to reproduce the authors' run.
"""

import numpy as np


def diffusion_operator(W, alpha):
    """Eq. 5–6. Returns the symmetric K."""
    d = W.sum(1)
    d_a = d ** -alpha
    return d_a[:, None] * W * d_a[None, :]


def canonical_sign(v):
    """Flip v so its largest-magnitude entry is positive (eigenvector signs are arbitrary)."""
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


def pg1_paper(W, alpha):
    """Eq. 5–8. Returns (g, eigenvalues in descending order)."""
    K = diffusion_operator(W, alpha)
    lam, V = np.linalg.eigh(K)                    # ascending
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    g = V[:, 1] / np.linalg.norm(V[:, 1])
    return canonical_sign(g), lam


def pg1_legacy(W, alpha, t=0.5, eps=1e-10):
    """Original pg_ext_2.diffusion_pg1_safe (minus its NaN/zero guards, which W here never needs)."""
    K = diffusion_operator(W, alpha)
    M = K / (K.sum(1, keepdims=True) + eps)
    lam, V = np.linalg.eigh(M)                    # assumes symmetric; reads the lower triangle only
    order = np.argsort(lam)[::-1]
    return (lam[order[1]] ** t) * V[:, order[1]], lam[order]


def pg1(W, alpha, mode="paper"):
    return {"paper": pg1_paper, "legacy": pg1_legacy}[mode](W, alpha)
