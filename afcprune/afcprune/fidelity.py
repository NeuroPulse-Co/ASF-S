"""
fidelity.py – AFC structure of a layer and how much a modified network preserves it (plan S2).

Layers:
  "out"         output logits (the seed paper's AFC layer)
  conv names    hidden channels: post-ReLU channel activation, global-average-pooled over H×W
                (one value per channel per sample), so AFC is a channel × channel graph

collect(model, loader, layers)      activations of several layers in one pass + labels + predictions
groups(labels, correct, ...)        the S sample indices per (class, correct|incorrect) group (pipeline.draw)
structure(acts, idx, sigma, ...)    per-group A, W, K, top-k diffusion embedding, PG1, the aligned contrast ΔG
                                    and its PCA, for one layer
compare(ref, other)                 similarity of two structures computed on the SAME groups (paired)

Groups are fixed by the *reference* (dense) model's correctness, and σ is fixed to the reference value, so a
comparison measures how the network's function changed, not sampling noise. `classes` restricts everything to
a class subset (one TIL task).
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from . import afc, config as C, contrast
from .pipeline import SPLITS, draw


@torch.no_grad()
def collect(model, loader, layers=("out",), max_batches=None):
    """Returns ({layer: (N, units) float64}, labels (N,), preds (N,)). Model is not modified."""
    m = copy.deepcopy(model).eval()
    conv_layers = [l for l in layers if l != "out"]
    bn_of = {c: bn for c, bn, _ in m.CONV_SEQ}
    store, hooks = {}, []
    for c in conv_layers:
        hooks.append(getattr(m, bn_of[c]).register_forward_hook(
            lambda _m, _i, out, c=c: store.__setitem__(c, F.relu(out).mean((2, 3)))))
    acts = {l: [] for l in layers}
    ys, preds = [], []
    try:
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            logits = m(x)
            if "out" in layers:
                acts["out"].append(logits.double().cpu())
            for c in conv_layers:
                acts[c].append(store[c].double().cpu())
            ys.append(y.cpu()); preds.append(logits.argmax(1).cpu())
    finally:
        for h in hooks:
            h.remove()
    return ({l: torch.cat(v).numpy() for l, v in acts.items()},
            torch.cat(ys).numpy(), torch.cat(preds).numpy())


def groups(labels, correct, classes=None, S=C.S_SAMPLES, seed=C.SEED):
    """{(class, split): sorted sample indices}, S per group (fewer if the pool is smaller)."""
    rng = np.random.default_rng(seed)
    classes = range(C.NUM_CLASSES) if classes is None else classes
    return {(c, s): draw(labels, correct, c, s, S, rng) for c in classes for s in SPLITS}


def _live(acts, idx_all, tol=1e-12):
    """Units with non-zero activation somewhere in the used samples (dead units are excluded: an all-zero
    unit is an isolated node with eigenvalue ~1 that can take over PG1; mnist/ fix #4)."""
    return np.abs(acts[idx_all]).max(0) > tol


def structure(acts, idx, sigma=None, eps=C.EPS, alpha=C.ALPHA, k_emb=C.SUBSPACE_K, k_pca=C.PCA_K, units=None):
    """
    acts (N, U) of one layer; idx from groups(). `units`: bool mask of units to include (default: live units
    of these samples). Returns dict with per-group K, embedding E, PG1 g, plus dG, comps, sigma, units.
    """
    keys = sorted(idx)
    if units is None:
        units = _live(acts, np.concatenate([idx[k] for k in keys]))
    X = acts[:, units]
    A = {k: afc.cosine_similarity(afc.activation_matrix(X[idx[k]])) for k in keys}
    sig = afc.median_sigma(list(A.values())) if sigma is None else float(sigma)
    out = dict(keys=keys, units=units, sigma=sig, K={}, E={}, g={}, lam={})
    for k in keys:
        W = afc.gaussian_affinity(A[k], sig, eps)
        lam, V = afc.spectrum(W, alpha)
        out["K"][k] = afc.diffusion_operator(W, alpha)
        out["E"][k] = V[:, 1:k_emb + 1]
        out["g"][k] = afc.canonical_sign(V[:, 1] / np.linalg.norm(V[:, 1]))
        out["lam"][k] = lam
    classes = sorted({c for c, _ in keys})
    g_corr = np.array([out["g"][(c, "correct")] for c in classes])
    g_inc = np.array([out["g"][(c, "incorrect")] for c in classes])
    out["dG"], _ = contrast.contrast_matrix(g_corr, g_inc)
    k_eff = max(1, min(k_pca, len(classes) - 1))
    out["comps"], out["ratio"], _ = contrast.pca(out["dG"], k_eff)
    return out


def compare(ref, other):
    """
    Paired similarity of two structures on the same groups and units (1 = identical function structure):
      contrast_sim   subspace similarity of the PCA components of ΔG (the paper's class-shared directions)
      embed_sim      mean over groups of subspace similarity of the top-k diffusion embeddings
      pg1_cos        mean |cos| of per-group PG1
      K_dist         mean relative Frobenius distance of per-group diffusion operators (0 = identical)
    """
    keys = ref["keys"]
    embed = np.mean([afc.subspace_similarity(ref["E"][k], other["E"][k]) for k in keys])
    pg1 = np.mean([abs(float(ref["g"][k] @ other["g"][k])) for k in keys])
    kd = np.mean([np.linalg.norm(ref["K"][k] - other["K"][k]) / (np.linalg.norm(ref["K"][k]) + 1e-12)
                  for k in keys])
    cs = afc.subspace_similarity(ref["comps"].T, other["comps"].T)
    return dict(contrast_sim=float(cs), embed_sim=float(embed), pg1_cos=float(pg1), K_dist=float(kd))


def logit_kl(ref_logits, logits):
    """Mean KL(p_ref || p) over samples: the output-matching twin (plan P1)."""
    p = torch.log_softmax(torch.as_tensor(ref_logits), 1)
    q = torch.log_softmax(torch.as_tensor(logits), 1)
    return float((p.exp() * (p - q)).sum(1).mean())


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019) between two (N, D) representations of the same samples."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))
