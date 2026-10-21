"""
pg.py - STAGE 2: artificial functional connectivity (paper Eq. 2-3) and the first
principal gradient PG1 (paper Eq. 5-8), per class and per correct/incorrect group.

Pipeline for one (class c, group in {correct, incorrect}, layer l)
------------------------------------------------------------------
  X  (units x samples)                        rows = unit "time series" (Eq. 1)
  A_ij = cos(x_i, x_j)                         Eq. 2   unit-unit similarity
  W_ij = exp(-(1 - A_ij)^2 / (2 sigma^2))      Eq. 3   Gaussian affinity (the AFC map)
  D = diag(W 1),  K = D^-a W D^-a              Eq. 5-6 diffusion normalisation
  K v_k = lambda_k v_k,  PG1 = v_2             Eq. 7-8 first NON-trivial eigenvector

Intuition: W is a graph whose nodes are units, with strong edges between units that
fire for the same images. v_2 of the normalised graph is the smoothest non-constant
function on it (the idea behind spectral clustering): units that are strongly
connected get similar PG1 values, so PG1 orders units along the dominant axis of
co-activation - the ANN analogue of the brain's unimodal-to-transmodal gradient.

Two variants are produced
-------------------------
  paper     Eq. 1-8 as written: up to S_MAX samples per group, post-ReLU activations,
            eigendecomposition of the symmetric K, plus two corrections the equations
            need in practice (dead units excluded; eigenvector signs fixed, see below).
  as_coded  a faithful replica of min_scripts/pg_ext_2.py, which is what produced the
            paper's numbers: ONE sample per group, pre-ReLU values, row-normalised M
            passed to eigh, v_2 scaled by lambda_2^t, no sign handling.

Output: outputs/pg/<variant>_<layer>.npz, consumed by importance.py.

Run:  python pg.py            (all variants and layers)
      python pg.py --plot     (also save contrast-vector plots to results/)
"""

import argparse
import os

import numpy as np
from scipy.linalg import eigh

import config

MIN_SAMPLES = 3   # below this a group's cosine matrix says almost nothing; skip the class


# ══════════════════════════════════════════════════════════════════════════════
# Paper variant
# ══════════════════════════════════════════════════════════════════════════════

def unit_affinity(X, sigma=config.SIGMA):
    """Eq. 2-3. X: (units, samples). Returns (W over ACTIVE units, active mask).

    A unit whose activations are all zero on these samples (a dead ReLU) has no
    direction, so its cosine with anything is undefined. If we kept it (e.g. treating
    the cosine as 0), it would become an almost isolated node: K_ii ~ 1 with ~0 edges,
    which creates an eigenvalue ~1 whose eigenvector sits on that single unit - and PG1
    would degenerate into a 'which units are dead' indicator. So dead units are left
    out of the graph here and given PG1 = 0 by the caller."""
    norms = np.linalg.norm(X, axis=1)
    active = norms > 1e-8
    Xn = X[active] / norms[active, None]            # unit-length rows
    A = np.clip(Xn @ Xn.T, -1.0, 1.0)                # Eq. 2: cosine similarity
    W = np.exp(-(1.0 - A) ** 2 / (2.0 * sigma ** 2)) # Eq. 3: Gaussian kernel
    W = 0.5 * (W + W.T) + 1e-6 * np.eye(len(W))      # Eq. 3: symmetrise + eps*I
    return W, active


def principal_gradient(W, alpha=config.ALPHA):
    """Eq. 5-8. Returns (PG1 as a unit vector, eigenvalues in descending order).

    K = D^-a W D^-a is SYMMETRIC, so scipy.linalg.eigh (symmetric solver) is the
    right tool. The original code instead row-normalised K into M (not symmetric)
    and still called eigh, which silently reads only one triangle of M; that happens
    to be close here, but it is not what Eq. 7 says."""
    d = W.sum(axis=1)                               # Eq. 5: degree of each unit
    d_pow = d ** (-alpha)
    K = d_pow[:, None] * W * d_pow[None, :]         # Eq. 6: D^-a W D^-a
    evals, evecs = eigh(K)                          # Eq. 7 (ascending order)
    evals, evecs = evals[::-1], evecs[:, ::-1]      # -> descending
    g = evecs[:, 1]                                 # Eq. 8: v_2, first non-trivial
    return g / np.linalg.norm(g), evals


def canonical_sign(g):
    """An eigenvector is only defined up to sign: if K v = lambda v then K(-v) = lambda(-v).
    Fix the convention 'the largest-magnitude entry is positive' so that PG1 vectors of
    different classes are comparable."""
    i = int(np.argmax(np.abs(g)))
    return g if g[i] >= 0 else -g


def pg1_paper(X, sigma, alpha):
    """PG1 for one group, scattered back to all units (dead units -> 0)."""
    W, active = unit_affinity(X, sigma)
    g_active, evals = principal_gradient(W, alpha)
    g = np.zeros(X.shape[0])
    g[active] = g_active
    return canonical_sign(g), evals, int(active.sum())


# ══════════════════════════════════════════════════════════════════════════════
# as_coded variant: replica of min_scripts/pg_ext_2.py
# ══════════════════════════════════════════════════════════════════════════════

def pg1_as_coded(x, sigma, alpha=0.5, t=0.5, eps=1e-10, jitter=1e-6):
    """x: activations of ONE sample, shape (units,). Mirrors pg_ext_2.py lines 33-101.

    With a single sample every unit is a 1-D 'vector', so its cosine with any other unit
    is +1 (same sign), -1 (opposite sign) or 0 (a zero unit). PG1 therefore reduces to
    the sign pattern of this one image's activations (engineering-log finding #3)."""
    X = x[:, None]                                            # (units, 1)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = np.divide(X, n, out=np.zeros_like(X), where=n > 0)   # sklearn: zero rows -> 0
    A = Xn @ Xn.T
    A = np.exp(-(1 - A) ** 2 / (2 * sigma ** 2)) + 1e-6
    A = (A + A.T) / 2
    A[np.diag_indices_from(A)] += 1e-10
    # diffusion_pg1_safe:
    A = np.where(A == 0, jitter, A)
    diag_pow = np.power(A.sum(axis=1), alpha)
    diag_pow = np.where(diag_pow == 0, eps, diag_pow)
    K = A / np.outer(diag_pow, diag_pow)
    M = K / (K.sum(axis=1, keepdims=True) + eps)              # row-stochastic, NOT symmetric
    evals, evecs = eigh(M)                                    # (as in the original)
    order = np.argsort(evals)[::-1]
    return (evals[order[1]] ** t) * evecs[:, order[1]], evals[order]


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════

def compute_layer(data, variant, layer, s_max, sigma, alpha, seed):
    """Compute PG1_correct, PG1_incorrect and the contrast for every class."""
    rng = np.random.default_rng(seed)
    labels, correct = data["labels"], data["correct"]
    # paper: post-ReLU (or logits for fc3). as_coded: pre-ReLU, as the original hooks saw.
    key = layer if variant == "paper" or layer == "fc3" else f"{layer}_pre"
    acts = data[key]
    n_units = acts.shape[1]

    out = {k: [] for k in ("g_corr", "g_incorr", "n_corr", "n_incorr",
                           "active_corr", "active_incorr", "evals_corr", "n_unit_eigs",
                           "flipped")}
    valid = []
    for c in range(config.NUM_CLASSES):
        idx_c = np.where((labels == c) & correct)[0]
        idx_i = np.where((labels == c) & ~correct)[0]
        # Placeholders for a skipped class; overwritten below when the class is valid.
        g_c = g_i = np.zeros(n_units); ev_c = np.zeros(3); act_c = act_i = 0; flipped = False

        if variant == "paper":
            # Up to s_max samples per group, drawn at random. MNIST has only 3-12
            # misclassified analysis samples per class, so 'incorrect' uses all of them.
            idx_c = rng.choice(idx_c, size=min(s_max, len(idx_c)), replace=False)
            idx_i = rng.choice(idx_i, size=min(s_max, len(idx_i)), replace=False)
            ok = len(idx_c) >= MIN_SAMPLES and len(idx_i) >= MIN_SAMPLES
            if ok:
                # Eq. 1: rows = units, columns = samples  ->  transpose of (samples, units)
                g_c, ev_c, act_c = pg1_paper(acts[idx_c].T, sigma, alpha)
                g_i, _, act_i = pg1_paper(acts[idx_i].T, sigma, alpha)
                # Align incorrect to correct: v and -v are the same eigenvector, and
                # g_c - g_i vs g_c + g_i are ORTHOGONAL for unit vectors, so without this
                # the contrast direction would be decided by a numerical coin flip.
                flipped = bool(g_c @ g_i < 0)
                if flipped:
                    g_i = -g_i
        else:
            # as_coded: one random sample per group (pg_ext_2.py shuffles, takes the 1st).
            ok = len(idx_c) >= 1 and len(idx_i) >= 1
            if ok:
                idx_c, idx_i = rng.choice(idx_c, 1), rng.choice(idx_i, 1)
                g_c, ev_c = pg1_as_coded(acts[idx_c[0]], config.AS_CODED_SIGMA)
                g_i, _ = pg1_as_coded(acts[idx_i[0]], config.AS_CODED_SIGMA)
                act_c = act_i = n_units
                flipped = False

        valid.append(ok)
        out["g_corr"].append(g_c); out["g_incorr"].append(g_i)
        out["n_corr"].append(len(idx_c)); out["n_incorr"].append(len(idx_i))
        out["active_corr"].append(act_c); out["active_incorr"].append(act_i)
        out["evals_corr"].append(ev_c[:3]); out["flipped"].append(flipped)
        # Number of eigenvalues ~1 = number of (near-)disconnected pieces of the graph.
        # If > 1, the top eigenvalue is repeated and PG1 is not uniquely defined.
        out["n_unit_eigs"].append(int(np.count_nonzero(ev_c > 0.999)) if ok else 0)

    res = {k: np.array(v) for k, v in out.items()}
    res["valid"] = np.array(valid)
    res["delta"] = res["g_corr"] - res["g_incorr"]     # Eq. 9, one row per class = Eq. 10
    return res


def report(variant, layer, r):
    """Diagnostics for the CORRECT-group graph of each class. What to look for:
      distinct PG1 vals : 2-3 means PG1 collapsed to a sign/alive pattern (S=1 symptom)
      pieces            : eigenvalues ~1; must be 1, otherwise PG1 is not unique
      gap               : lam2 - lam3; the larger, the more stable PG1 is to resampling
      flipped           : whether g_incorr had to be sign-flipped to align with g_corr"""
    print(f"\n── {variant} / {layer} ──")
    print(" cls | n_corr n_inc | active units | distinct PG1 vals | pieces | lam2   gap   | flipped")
    for c in range(config.NUM_CLASSES):
        if not r["valid"][c]:
            print(f"  {c}  | skipped (too few samples)"); continue
        distinct = len(np.unique(np.round(r["g_corr"][c], 6)))
        _, l2, l3 = r["evals_corr"][c]
        print(f"  {c}  |  {r['n_corr'][c]:3d}  {r['n_incorr'][c]:3d}  |  {r['active_corr'][c]:3d} / {r['active_incorr'][c]:3d}"
              f"   |  {distinct:4d}             |   {r['n_unit_eigs'][c]:2d}   | {l2:.3f}  {l2 - l3:.3f} | {r['flipped'][c]}")
    bad = [c for c in range(config.NUM_CLASSES) if r["valid"][c] and r["n_unit_eigs"][c] > 1]
    if bad and variant == "as_coded":
        print("  NOTE: disconnected graph is inherent to the original method - one sample "
              "splits units into a same-sign block and an opposite-sign block.")
    elif bad:
        print(f"  WARNING: graph disconnected for classes {bad} -> PG1 not unique there "
              f"(try a larger --sigma)")


def plot_contrast(variant, layer, r, path):
    """One line per class: the contrast vector delta_g (compare paper Fig. 20-23)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in np.where(r["valid"])[0]:
        ax.plot(r["delta"][c], lw=1, label=f"class {c}")
    ax.set_xlabel(f"{layer} unit index"); ax.set_ylabel("PG1_corr - PG1_incorr")
    ax.set_title(f"Contrast vectors ({variant}, {layer})")
    ax.legend(ncol=5, fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["paper:fc2", "paper:fc3", "as_coded:fc2"],
                    help="variant:layer pairs to compute")
    ap.add_argument("--s_max", type=int, default=config.S_MAX)
    ap.add_argument("--sigma", type=float, default=config.SIGMA)
    ap.add_argument("--alpha", type=float, default=config.ALPHA)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    data = dict(np.load(config.ACTIVATIONS_PATH))
    for run in args.runs:
        variant, layer = run.split(":")
        r = compute_layer(data, variant, layer, args.s_max, args.sigma, args.alpha, args.seed)
        report(variant, layer, r)
        path = os.path.join(config.PG_DIR, f"{variant}_{layer}.npz")
        # **r keys are fixed result names, never `allow_pickle` (Pylance false positive).
        np.savez(path, sigma=args.sigma, alpha=args.alpha, s_max=args.s_max, **r)  # pyright: ignore[reportArgumentType]
        print(f"Saved -> {path}")
        if args.plot:
            png = os.path.join(config.RESULTS_DIR, f"pg_contrast_{variant}_{layer}.png")
            plot_contrast(variant, layer, r, png)
            print(f"Plot  -> {png}")


if __name__ == "__main__":
    main()
