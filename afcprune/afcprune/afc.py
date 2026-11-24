"""
afc.py – Artificial Functional Connectivity and its diffusion embedding [paper §3–4, Eq. 1–8].

  X ∈ R^{N×S}: unit × sample activations                          (Eq. 1)
  A_ij = cos(x_i, x_j)                                             (Eq. 2)
  W_ij = exp(−(1 − A_ij)² / 2σ²),  W ← ½(W + Wᵀ) + εI              (Eq. 3)
  K = D^-α W D^-α,  D = diag(W1)                                   (Eq. 5–6)
  K v_k = λ_k v_k;  PG1 = v_2 (unit length)                        (Eq. 7–8)

Ported from cifar10/asfs/{afc,pg}.py ("paper" mode only; the legacy mode lives in the golden tests).
Additions for afcprune (plan P3):
  embedding()       the top-k non-trivial eigenvectors, not only v_2, so fidelity can compare subspaces,
                    which is stable when λ2 ≈ λ3 swaps the order of v_2 and v_3
  principal_cos()   cosines of the principal angles between two subspaces
  bootstrap_pg1()   resampling stability of PG1 (|cos| to the full-sample PG1)
"""

import numpy as np


# ── Eq. 1–3 ───────────────────────────────────────────────────

def activation_matrix(acts):
    """Eq. 1. acts: (S, N) samples × units → X: (N, S)."""
    return np.asarray(acts, dtype=np.float64).T


def cosine_similarity(X, eps=1e-12):
    """Eq. 2. X: (N, S) → A: (N, N)."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


def gaussian_affinity(A, sigma, eps):
    """Eq. 3."""
    W = np.exp(-(1.0 - A) ** 2 / (2.0 * sigma ** 2))
    return 0.5 * (W + W.T) + eps * np.eye(len(W))


def median_sigma(As):
    """Median of off-diagonal (1 − A) pooled over a list of cosine matrices."""
    off = [(1.0 - A)[~np.eye(len(A), dtype=bool)] for A in As]
    return float(np.median(np.concatenate(off)))


def resolve_sigma(sigma, As):
    return median_sigma(As) if sigma == "median" else float(sigma)


def afc(acts, sigma, eps):
    """acts (S, N) → (A, W)."""
    A = cosine_similarity(activation_matrix(acts))
    return A, gaussian_affinity(A, sigma, eps)


def connectivity(W, thresh=0.01):
    """Fraction of off-diagonal W above `thresh` (≈ 0 means W ≈ εI, no edges)."""
    off = W[~np.eye(len(W), dtype=bool)]
    return float((off > thresh).mean())


# ── Eq. 5–8 ───────────────────────────────────────────────────

def diffusion_operator(W, alpha):
    d = W.sum(1)
    d_a = d ** -alpha
    return d_a[:, None] * W * d_a[None, :]


def canonical_sign(v):
    """Flip so the largest-|·| entry is positive."""
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


def spectrum(W, alpha):
    """Eigen-decomposition of K, descending. Returns (λ (N,), V (N, N) columns = eigenvectors)."""
    lam, V = np.linalg.eigh(diffusion_operator(W, alpha))
    order = np.argsort(lam)[::-1]
    return lam[order], V[:, order]


def pg1(W, alpha):
    """Eq. 5–8. Returns (g unit-norm with canonical sign, λ descending)."""
    lam, V = spectrum(W, alpha)
    g = V[:, 1] / np.linalg.norm(V[:, 1])
    return canonical_sign(g), lam


def embedding(W, alpha, k):
    """Top-k non-trivial eigenvectors (v_2 … v_{k+1}) as an (N, k) orthonormal basis, plus λ."""
    lam, V = spectrum(W, alpha)
    return V[:, 1:k + 1], lam


# ── Comparing embeddings (plan P3) ────────────────────────────

def principal_cos(U, V):
    """Cosines of the principal angles between span(U) and span(V); U, V (N, k) orthonormal columns."""
    return np.clip(np.linalg.svd(U.T @ V, compute_uv=False), 0.0, 1.0)


def subspace_similarity(U, V):
    """Mean squared principal cosine in [0, 1] (1 = same subspace); sign- and order-invariant."""
    return float(np.mean(principal_cos(U, V) ** 2))


def bootstrap_pg1(acts, sigma, eps, alpha, n_boot=50, seed=0):
    """
    PG1 stability under resampling of the S samples (acts (S, N)).
    Returns |cos| between each bootstrap PG1 and the full-sample PG1, shape (n_boot,).
    """
    rng = np.random.default_rng(seed)
    g0, _ = pg1(afc(acts, sigma, eps)[1], alpha)
    S = len(acts)
    out = np.empty(n_boot)
    for b in range(n_boot):
        g, _ = pg1(afc(acts[rng.integers(0, S, S)], sigma, eps)[1], alpha)
        out[b] = abs(float(g @ g0))
    return out
