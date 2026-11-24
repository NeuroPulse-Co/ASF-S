"""
data_driven.py – activation-based filter-importance baselines (higher = keep).

All use the channel output a = ReLU(BN(conv(x))) (Conv2Net/Conv6Net apply ReLU right after BN), captured with
a forward hook on the BN module of a deepcopy of the model, over ctx.calib (normalised, un-augmented batches).

  taylor     first-order Taylor (Molchanov et al. 2017/2019): per sample |Σ_hw a · ∂L/∂a| with L = CE,
             averaged over samples. ∂L/∂a is read from the BN output z: a · ∂L/∂z = a · ∂L/∂a since a = 0
             wherever z ≤ 0.
  hrank      mean matrix rank of each channel's H×W feature map (Lin et al. 2020), over
             ctx.extra.get("hrank_batches", 5) batches
  act_mean   mean |a| over samples and positions

The model is put in eval mode (BN uses running statistics) and is never modified.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from . import register


def _hooked_copy(model, device, keep_grad):
    m = copy.deepcopy(model).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    store = {}

    def make(name):
        def hook(_mod, _inp, out):
            if keep_grad:
                out.retain_grad()
            store[name] = out
        return hook

    for conv, bn, _ in m.CONV_SEQ:
        getattr(m, bn).register_forward_hook(make(conv))
    return m, store


def _batches(ctx, limit=None):
    if ctx.calib is None:
        raise ValueError("this scorer needs ctx.calib (an iterable of (x, y) batches)")
    for i, (x, y) in enumerate(ctx.calib):
        if limit is not None and i >= limit:
            break
        yield x, y


@register("taylor")
def taylor(model, ctx):
    torch.manual_seed(ctx.seed)
    m, store = _hooked_copy(model, ctx.device, keep_grad=True)
    tot, n = {}, 0
    for x, y in _batches(ctx, ctx.extra.get("taylor_batches")):
        x, y = x.to(ctx.device).requires_grad_(True), y.to(ctx.device)   # input grad so BN outputs need grad
        loss = F.cross_entropy(m(x), y, reduction="sum")
        loss.backward()
        for c, z in store.items():
            s = (F.relu(z) * z.grad).sum((2, 3)).abs().sum(0).detach().double().cpu()
            tot[c] = tot.get(c, 0) + s
        n += len(y)
    return {c: (v / n).numpy() for c, v in tot.items()}


@register("hrank")
@torch.no_grad()
def hrank(model, ctx):
    m, store = _hooked_copy(model, ctx.device, keep_grad=False)
    tot, n = {}, 0
    for x, _ in _batches(ctx, ctx.extra.get("hrank_batches", 5)):
        m(x.to(ctx.device))
        for c, z in store.items():
            r = torch.linalg.matrix_rank(F.relu(z).double()).sum(0).cpu()      # (B, C, H, W) → (C,)
            tot[c] = tot.get(c, 0) + r
        n += len(x)
    return {c: (v.double() / n).numpy() for c, v in tot.items()}


@register("act_mean")
@torch.no_grad()
def act_mean(model, ctx):
    m, store = _hooked_copy(model, ctx.device, keep_grad=False)
    tot, n = {}, 0
    for x, _ in _batches(ctx, ctx.extra.get("act_batches")):
        m(x.to(ctx.device))
        for c, z in store.items():
            tot[c] = tot.get(c, 0) + F.relu(z).double().mean((2, 3)).sum(0).cpu()
        n += len(x)
    return {c: (v / n).numpy() for c, v in tot.items()}
