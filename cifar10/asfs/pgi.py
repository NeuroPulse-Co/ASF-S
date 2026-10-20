"""
pgi.py – Principal Gradient Importance per conv filter [paper §4, Eq. 12–13].

  s_ik  = ⟨ w_i , u_k ⊙ w̄^(f) ⟩                     (Eq. 12)
  PGI_i = N( (1/K) Σ_k |s_ik| ),  N = min–max per layer   (Eq. 13)

u_k ∈ R^C lives on the output units (C = 10), so u_k ⊙ w̄^(f) needs w̄^(f) ∈ R^C, and then
w_i must be in R^C too. A conv filter's flattened weights are not (27 or 288 values), and the
paper does not say how they are mapped. Two modes:

  literal   what the original min_scripts/pg_pruning_conv2.py computes:
              w̄ = mean of the FC weight over its *rows* (length = FC inputs, 16384),
              all three vectors truncated to their common length (10), so only the first
              10 kernel weights of each filter are used. Kept to reproduce the authors' run.

  aligned   dimensionally consistent:
              w̄^(f) = mean of the output weight over its *columns*: one value per output unit;
              w_i is replaced by e_i ∈ R^C, filter i's effective linear weight to each
              output unit. For the last conv layer, e_i is the mean of the output-layer
              columns that channel i feeds (one per spatial position), times its BN scale.
              For earlier layers, e is propagated back through the next conv's
              kernel-summed weights: E_ℓ = a_ℓ ⊙ (M_{ℓ+1}ᵀ E_{ℓ+1}), M[i, j] = Σ_kernel W[i, j].
              ReLU and pooling are ignored, so e_i is a linear surrogate of filter i's
              structural path to the output.

Eq. 13 takes mean_k |s_ik|. The original code took |mean_k s_ik|; abs_of_mean=True reproduces it.
"""

import numpy as np
import torch


def minmax(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def scores(Wvec, targets, abs_of_mean=False):
    """Wvec (F, D) unit vectors, targets (K, D) → PGI (F,) per Eq. 12–13."""
    s = Wvec @ targets.T                                          # (F, K): s_ik
    imp = np.abs(s.mean(1)) if abs_of_mean else np.abs(s).mean(1)
    return minmax(imp)


def output_weight(model):
    """Linear map from the last conv's flattened output to the output units (C, D_in).
    Conv2Net: fc. Conv6Net: fc2 @ fc1 (ReLU and dropout ignored)."""
    Ws = [getattr(model, n).weight.detach().cpu().double().numpy() for n in model.FC_LAYERS]
    out = Ws[-1]
    for W in reversed(Ws[:-1]):
        out = out @ W
    return out


def bn_scale(bn):
    return (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().cpu().double().numpy()


def effective_output_vectors(model):
    """{conv name: E (F, C)}: each filter's effective linear weight to every output unit."""
    Wout = output_weight(model)                                   # (C, F_last * hw)
    seq = model.CONV_SEQ
    last_conv, last_bn, _ = seq[-1]
    F_last = getattr(model, last_conv).out_channels
    hw = Wout.shape[1] // F_last
    E = {}
    e = Wout.reshape(Wout.shape[0], F_last, hw).mean(2).T         # (F_last, C): channel-major flatten
    E[last_conv] = bn_scale(getattr(model, last_bn))[:, None] * e
    for (conv, bn, nxt) in reversed(seq[:-1]):
        Wn = getattr(model, nxt).weight.detach().cpu().double().numpy()   # (F_next, F, kh, kw)
        M = Wn.sum((2, 3))                                                # (F_next, F)
        E[conv] = bn_scale(getattr(model, bn))[:, None] * (M.T @ E[nxt])
    return E


def pgi_aligned(model, components, abs_of_mean=False):
    """{conv name: PGI (F,)} in aligned mode. components: (K, C)."""
    w_bar = output_weight(model).mean(1)                          # (C,)
    targets = components * w_bar[None, :]                         # u_k ⊙ w̄^(f)
    return {name: scores(E, targets, abs_of_mean) for name, E in effective_output_vectors(model).items()}


def pgi_literal(model, components, abs_of_mean=True):
    """{conv name: PGI (F,)}, the original min_scripts formula (prefix truncation)."""
    fc_w = getattr(model, model.FC_LAYERS[-1]).weight.detach().cpu().double().numpy()
    avg_fc = fc_w.mean(0)                                         # mean over rows (original axis=0)
    out = {}
    for conv, _, _ in model.CONV_SEQ:
        w = getattr(model, conv).weight.detach().cpu().double().numpy()
        flat = w.reshape(w.shape[0], -1)
        sz = min(flat.shape[1], avg_fc.shape[0], components.shape[1])
        targets = components[:, :sz] * avg_fc[None, :sz]
        out[conv] = scores(flat[:, :sz], targets, abs_of_mean)
    return out


def pgi(model, components, mode="aligned", abs_of_mean=False):
    if mode == "aligned":
        return pgi_aligned(model, components, abs_of_mean)
    if mode == "literal":
        return pgi_literal(model, components, abs_of_mean)
    raise ValueError(mode)
