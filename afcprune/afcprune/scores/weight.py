"""
weight.py – data-free filter-importance baselines (higher = keep).

  l1, l2     ‖W_f‖₁, ‖W_f‖₂ of each conv filter (as cifar10 step 8)
  random     uniform noise, np.random.default_rng(ctx.seed)
  bn_gamma   |γ| of the conv's BatchNorm (network slimming criterion, without sparsity training)
  fpgm       Σ_j ‖W_f − W_j‖₂ over the layer's other filters (He et al. 2019): filters near the geometric
             median are replaceable, so a small distance sum = low importance
"""

import numpy as np
import torch

from . import conv_names, register


def _weights(model):
    return {c: getattr(model, c).weight.detach().cpu().double() for c in conv_names(model)}


@register("l1")
def l1(model, ctx):
    return {c: w.abs().sum((1, 2, 3)).numpy() for c, w in _weights(model).items()}


@register("l2")
def l2(model, ctx):
    return {c: w.pow(2).sum((1, 2, 3)).sqrt().numpy() for c, w in _weights(model).items()}


@register("random")
def random(model, ctx):
    rng = np.random.default_rng(ctx.seed)
    return {c: rng.random(getattr(model, c).out_channels) for c in conv_names(model)}


@register("bn_gamma")
def bn_gamma(model, ctx):
    return {c: getattr(model, bn).weight.detach().cpu().double().abs().numpy() for c, bn, _ in model.CONV_SEQ}


@register("fpgm")
def fpgm(model, ctx):
    out = {}
    for c, w in _weights(model).items():
        flat = w.reshape(w.shape[0], -1)
        out[c] = torch.cdist(flat, flat).sum(1).numpy()
    return out
