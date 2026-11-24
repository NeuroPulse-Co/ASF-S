"""
pipeline.py – the output-layer AFC pipeline in one call (cifar10 steps 2–5, same sampling).

output_contrast(model, pg_loader) runs the model over the PG image set, draws S samples per
(class, correct|incorrect) group with the same seeded draw as cifar10/steps/step3_afc.py, builds A and W
(Eq. 1–3, σ by the median rule over all groups), PG1 per group (Eq. 5–8), the aligned contrast ΔG and its
PCA (Eq. 9–11). Diagnostics for plan P3 come with it: spectral gaps, min group size and bootstrap PG1
stability.
"""

import numpy as np
import torch

from . import afc, config as C, contrast

SPLITS = ("correct", "incorrect")


@torch.no_grad()
def output_activations(model, loader):
    """(acts (N, units), labels (N,), correct (N,)) of the output layer, eval mode, no augmentation."""
    model.eval()
    acts, labels = [], []
    for x, y in loader:
        acts.append(model(x).float().cpu()); labels.append(y.cpu())
    acts, labels = torch.cat(acts).numpy().astype(np.float64), torch.cat(labels).numpy()
    return acts, labels, acts.argmax(1) == labels


def draw(labels, correct, c, split, S, rng):
    """Same as cifar10/steps/step3_afc.draw."""
    pool = np.flatnonzero((labels == c) & (correct if split == "correct" else ~correct))
    return np.sort(rng.permutation(pool)[:S])


def contrast_from_acts(acts, labels, correct, S=C.S_SAMPLES, sigma=C.SIGMA, eps=C.EPS, alpha=C.ALPHA,
                       k=C.PCA_K, seed=C.SEED, n_boot=0):
    n_cls, n = C.NUM_CLASSES, acts.shape[1]
    rng = np.random.default_rng(seed)
    idx = [[draw(labels, correct, c, s, S, rng) for s in SPLITS] for c in range(n_cls)]
    counts = np.array([[len(i) for i in row] for row in idx])
    A = np.array([[afc.cosine_similarity(afc.activation_matrix(acts[i])) for i in row] for row in idx])
    sig = afc.resolve_sigma(sigma, A.reshape(-1, n, n))
    W = np.array([[afc.gaussian_affinity(A[c, s], sig, eps) for s in range(2)] for c in range(n_cls)])
    g = np.zeros((n_cls, 2, n)); lam = np.zeros((n_cls, 2, n))
    for c in range(n_cls):
        for s in range(2):
            g[c, s], lam[c, s] = afc.pg1(W[c, s], alpha)
    dG, flip = contrast.contrast_matrix(g[:, 0], g[:, 1])
    comps, ratio, _ = contrast.pca(dG, k)
    out = dict(idx=idx, counts=counts, A=A, W=W, sigma=sig, g=g, lam=lam, dG=dG, flip=flip,
               comps=comps, ratio=ratio,
               gap12_min=float((lam[..., 0] - lam[..., 1]).min()),
               gap23_min=float((lam[..., 1] - lam[..., 2]).min()),
               gap23_median=float(np.median(lam[..., 1] - lam[..., 2])),
               topk_explained=float(ratio[:k].sum()))
    if n_boot:
        boot = np.array([[np.median(afc.bootstrap_pg1(acts[idx[c][s]], sig, eps, alpha, n_boot, seed))
                          for s in range(2)] for c in range(n_cls)])
        out["boot_median_abs_cos"] = boot
    return out


def output_contrast(model, pg_loader, **kw):
    acts, labels, correct = output_activations(model, pg_loader)
    out = contrast_from_acts(acts, labels, correct, **kw)
    out["pg_accuracy"] = float(100 * correct.mean())
    return out
