"""
metrics.py – accuracy, parameter count, FLOPs and ECE.
"""

import torch
import torch.nn.functional as F


def count_params(model):
    """All parameters, frozen or not (ptflops skips frozen ones)."""
    return sum(p.numel() for p in model.parameters())


def count_flops(model, input_shape=(3, 32, 32)):
    """FLOPs = 2 × MACs from ptflops, the convention behind the paper's Table 4."""
    from ptflops import get_model_complexity_info
    macs, _ = get_model_complexity_info(model, input_shape, as_strings=False,
                                        print_per_layer_stat=False, verbose=False)
    return 2 * macs


@torch.no_grad()
def predict(model, loader):
    """Returns (logits, labels) over a whole loader."""
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(model(x)); ys.append(y)
    return torch.cat(out), torch.cat(ys)


def accuracy(logits, labels):
    return 100.0 * (logits.argmax(1) == labels).float().mean().item()


def ece(logits, labels, n_bins=15):
    """Expected calibration error with equal-width confidence bins (as the original scripts)."""
    conf, pred = F.softmax(logits, 1).max(1)
    total = torch.zeros((), device=logits.device)
    edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * ((pred[m] == labels[m]).float().mean() - conf[m].mean()).abs()
    return total.item()
