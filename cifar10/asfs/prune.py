"""
prune.py – physical structured pruning of conv filters [paper Eq. 14: prune PGI_i < τ].

prune(model, keep) returns a new, smaller model: for every conv in model.CONV_SEQ the kept
output filters are copied, together with their BN channels, the next conv's matching input
channels, and, for the last conv, the matching FC input columns (channel-major flatten:
column = channel · hw + position).

masked(model, keep) is the same network at full size with the dropped channels' *inputs to
the next layer* zeroed. The two compute the same function, which the tests check. Zeroing a
filter's own weights would not: BN + ReLU turn a zero map into a constant ReLU(β − γμ/σ)
that still reaches the next layer.
"""

import copy

import numpy as np
import torch
import torch.nn as nn


def keep_from_scores(score, tau):
    """Eq. 14: keep filters with PGI ≥ τ. At least one filter is kept (the best)."""
    keep = np.asarray(score) >= tau
    if not keep.any():
        keep[np.argmax(score)] = True
    return keep


def keep_top_k(score, k):
    """Keep the k highest-scoring filters (for baselines matched to a kept count)."""
    keep = np.zeros(len(score), bool)
    keep[np.argsort(-np.asarray(score), kind="stable")[:k]] = True
    return keep


def _idx(mask, device):
    return torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=device)


def _conv_like(old, in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, old.kernel_size, old.stride, old.padding, old.dilation,
                     old.groups, old.bias is not None).to(old.weight.device, old.weight.dtype)


@torch.no_grad()
def prune(model, keep):
    """keep: {conv name: bool array over its output filters}. Returns a pruned deep copy."""
    m = copy.deepcopy(model)
    for conv_name, bn_name, next_name in m.CONV_SEQ:
        conv = getattr(m, conv_name)
        dev = conv.weight.device
        idx = _idx(keep[conv_name], dev)
        orig_out = conv.out_channels

        new = _conv_like(conv, conv.in_channels, len(idx))
        new.weight.copy_(conv.weight[idx])
        if conv.bias is not None:
            new.bias.copy_(conv.bias[idx])
        setattr(m, conv_name, new)

        bn = getattr(m, bn_name)
        nbn = nn.BatchNorm2d(len(idx), bn.eps, bn.momentum, bn.affine, bn.track_running_stats).to(dev)
        nbn.weight.copy_(bn.weight[idx]); nbn.bias.copy_(bn.bias[idx])
        nbn.running_mean.copy_(bn.running_mean[idx]); nbn.running_var.copy_(bn.running_var[idx])
        nbn.num_batches_tracked.copy_(bn.num_batches_tracked)
        setattr(m, bn_name, nbn)

        if next_name is not None:
            nxt = getattr(m, next_name)
            nn_ = _conv_like(nxt, len(idx), nxt.out_channels)
            nn_.weight.copy_(nxt.weight[:, idx])
            if nxt.bias is not None:
                nn_.bias.copy_(nxt.bias)
            setattr(m, next_name, nn_)
        else:
            fc = getattr(m, m.FC_AFTER_CONV)
            hw = fc.in_features // orig_out
            cols = (idx[:, None] * hw + torch.arange(hw, device=dev)).reshape(-1)
            nfc = nn.Linear(len(cols), fc.out_features, fc.bias is not None).to(dev, fc.weight.dtype)
            nfc.weight.copy_(fc.weight[:, cols])
            if fc.bias is not None:
                nfc.bias.copy_(fc.bias)
            setattr(m, m.FC_AFTER_CONV, nfc)
    # New modules start in training mode; a train-mode BN normalises with batch statistics
    # instead of the copied running ones. Match the source model's mode.
    m.train(model.training)
    return m


@torch.no_grad()
def masked(model, keep):
    """Full-size copy computing the same function as prune(model, keep)."""
    m = copy.deepcopy(model)
    for conv_name, _, next_name in m.CONV_SEQ:
        drop = torch.as_tensor(~np.asarray(keep[conv_name]), device=next(m.parameters()).device)
        if next_name is not None:
            getattr(m, next_name).weight[:, drop] = 0
        else:
            fc = getattr(m, m.FC_AFTER_CONV)
            hw = fc.in_features // len(drop)
            fc.weight.view(fc.out_features, len(drop), hw)[:, drop] = 0
    return m


def kept_counts(keep):
    return {k: int(np.sum(v)) for k, v in keep.items()}
