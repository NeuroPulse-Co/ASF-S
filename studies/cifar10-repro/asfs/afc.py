"""
afc.py – Artificial Functional Connectivity [paper §3, Eq. 1–3].

X^(ℓ) ∈ R^{N_ℓ × S}: row i is unit i's activations over S samples (Eq. 1).
A_ij = cosine(x_i, x_j)                                            (Eq. 2)
W_ij = exp(−(1 − A_ij)² / 2σ²),  W ← ½(W + Wᵀ) + εI                 (Eq. 3)
"""

import numpy as np


def activation_matrix(acts):
    """Eq. 1. acts: (S, N) samples × units, as saved by step 2 → X: (N, S)."""
    return np.asarray(acts, dtype=np.float64).T


def cosine_similarity(X, eps=1e-12):
    """Eq. 2. X: (N, S) → A: (N, N), unit × unit cosine over samples."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


def gaussian_affinity(A, sigma, eps):
    """Eq. 3. A: (N, N) cosine → W: (N, N) symmetric weighted adjacency."""
    W = np.exp(-(1.0 - A) ** 2 / (2.0 * sigma ** 2))
    return 0.5 * (W + W.T) + eps * np.eye(len(W))


def median_sigma(As):
    """Median rule for σ: median of off-diagonal (1 − A) pooled over a list of cosine matrices."""
    off = [(1.0 - A)[~np.eye(len(A), dtype=bool)] for A in As]
    return float(np.median(np.concatenate(off)))


def resolve_sigma(sigma, As):
    """config.SIGMA may be a number or "median"."""
    return median_sigma(As) if sigma == "median" else float(sigma)


def afc(acts, sigma, eps):
    """acts (S, N) → (A, W)."""
    A = cosine_similarity(activation_matrix(acts))
    return A, gaussian_affinity(A, sigma, eps)


def connectivity(W, thresh=0.01):
    """Fraction of off-diagonal entries of W above `thresh`: near 0 means W ≈ εI, a graph with no edges."""
    off = W[~np.eye(len(W), dtype=bool)]
    return float((off > thresh).mean())
