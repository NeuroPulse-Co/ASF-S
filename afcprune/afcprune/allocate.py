"""
allocate.py – how many filters each layer keeps for a total-params↓ budget (plans P5, P8).

The seed paper picks per-layer PGI thresholds by a grid search judged on the test set. Here:
  1. candidates(model, budget, tol): every per-layer kept-count vector whose *total* params↓ (all layers,
     incl. BN and FC) is within ±tol points of the budget. Exhaustive for Conv2Net (32 × 64 combinations);
     for deeper models a uniform-fraction family plus single-layer perturbations.
  2. choose(...): pick the candidate with the best val score under a caller-supplied criterion
     `evaluate(keep) -> float`, the SAME criterion for every selector (P10). Only val/train data may be
     used inside `evaluate`; the protocol guard enforces that the test split is unreachable.

Kept filters are always the top-k of the selector's scores, so a selector is fully described by its scores
plus this allocation rule.
"""

import itertools

import numpy as np

from . import metrics, surgery


def conv_sizes(model):
    return {c: getattr(model, c).out_channels for c, _, _ in model.CONV_SEQ}


def count_params_kept(model, kept):
    """
    Analytic parameter count of the pruned model (no copy): each conv keeps kept[c] filters, its BN keeps
    2·kept[c] affine params, the next conv's input channels shrink to kept[c], and the FC right after the
    last conv keeps kept[last]·hw input columns. Equal to metrics.count_params(surgery.prune(...))
    (tests/test_allocate.py).
    """
    total = metrics.count_params(model)
    prev_in = None
    for conv_name, bn_name, _ in model.CONV_SEQ:
        conv, bn = getattr(model, conv_name), getattr(model, bn_name)
        kh, kw = conv.kernel_size
        cin = conv.in_channels if prev_in is None else prev_in
        old = conv.weight.numel() + (conv.bias.numel() if conv.bias is not None else 0)
        new = kept[conv_name] * cin * kh * kw + (kept[conv_name] if conv.bias is not None else 0)
        total += new - old
        if bn.affine:
            total += 2 * (kept[conv_name] - bn.num_features)
        prev_in = kept[conv_name]
    last = model.CONV_SEQ[-1][0]
    fc = getattr(model, model.FC_AFTER_CONV)
    hw = fc.in_features // getattr(model, last).out_channels
    total += (kept[last] - getattr(model, last).out_channels) * hw * fc.out_features
    return total


def params_red(model, kept, base=None):
    """Total params↓ (%) when each conv keeps kept[c] filters (which ones does not matter for the count)."""
    base = base or metrics.count_params(model)
    return 100.0 * (1 - count_params_kept(model, kept) / base)


def _family(model):
    sizes = conv_sizes(model)
    names = list(sizes)
    if len(names) <= 2:
        yield from (dict(zip(names, ks)) for ks in itertools.product(*(range(1, sizes[c] + 1) for c in names)))
        return
    seen = set()
    for f in np.linspace(0.02, 1.0, 50):
        base = {c: max(1, int(round(f * sizes[c]))) for c in names}
        for c in [None] + names:
            for d in ([0] if c is None else [-2, -1, 1, 2]):
                k = dict(base)
                if c is not None:
                    k[c] = int(np.clip(k[c] + d * max(1, sizes[c] // 16), 1, sizes[c]))
                t = tuple(k[n] for n in names)
                if t not in seen:
                    seen.add(t)
                    yield k


def candidates(model, budget, tol=0.5):
    """[(kept dict, params↓)] with |params↓ − budget| ≤ tol, sorted by distance to the budget."""
    base = metrics.count_params(model)
    out = []
    for kept in _family(model):
        r = params_red(model, kept, base)
        if abs(r - budget) <= tol:
            out.append((kept, r))
    return sorted(out, key=lambda kr: abs(kr[1] - budget))


def keep_from_counts(scores, kept):
    """Top-k of each layer's scores → {conv: bool mask}."""
    return {c: surgery.keep_top_k(scores[c], kept[c]) for c in kept}


def choose(scores, cands, evaluate, max_candidates=None):
    """
    scores: {conv: (F,)}; cands: output of candidates(); evaluate(keep) -> val score (higher is better).
    Returns (best kept dict, best params↓, table of (kept, params↓, score)).
    """
    table = []
    for kept, r in cands[:max_candidates]:
        table.append((kept, r, float(evaluate(keep_from_counts(scores, kept)))))
    best = max(table, key=lambda t: t[2])
    return best[0], best[1], table
