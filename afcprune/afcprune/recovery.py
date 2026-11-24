"""
recovery.py – recovering accuracy after structured pruning, as named tiers (plan P7).

  T0  none                         the pruned network as is
  T1  strict head-only             only the last FC layer changes: gradient FT with every BatchNorm frozen
                                   (finetune mode "fc_frozen") or closed-form ridge refit (ridge_head)
  T2  T1 + BN recalibration        BatchNorm running statistics re-estimated by forward passes
                                   (bn_recalibrate); no weight changes
  T3  T2 + LS compensation         next-layer conv weights refit in closed form so their pre-BN output
                                   reproduces the dense network's (ls_compensate); no gradients
  T4  full fine-tuning             every layer trainable (finetune mode "full"), literature reference

Note on run01's −10.50 (cifar10 step 8): it used finetune mode "fc", where only FC weights train but
BatchNorm is in train mode, so its running statistics re-adapt during fine-tuning. That is T1 + BN
adaptation (between T1 and T2), not strict T1.

Loaders are iterables of normalised (x, y) batches already on the right device. The closed-form steps
(ridge_head, bn_recalibrate, ls_compensate) should get an un-augmented train loader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .train import fit

TIERS = {
    "T0": "none: pruned network as is",
    "T1": "strict head-only: last FC only (fc_frozen gradient FT, or closed-form ridge), BN frozen",
    "T2": "T1 + BatchNorm running-statistics recalibration (forward passes only)",
    "T3": "T2 + closed-form least-squares compensation of next-layer conv weights",
    "T4": "full fine-tuning of every layer",
}

MODES = ("fc", "fc_frozen", "full")


def _bns(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def _last_fc(model):
    return getattr(model, model.FC_LAYERS[-1])


def _device(model):
    return next(model.parameters()).device


# ── gradient fine-tuning (T1 fc_frozen, "fc", T4 full) ────────

def configure(model, mode):
    """Set requires_grad for `mode`. Returns (trainable parameters, bn_eval)."""
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return list(model.parameters()), False
    for p in model.parameters():
        p.requires_grad = False
    params = []
    for name in model.FC_LAYERS:
        for p in getattr(model, name).parameters():
            p.requires_grad = True
            params.append(p)
    return params, mode == "fc_frozen"


def finetune(model, mode, train_loader, val_loader, lr, max_epochs, patience, log=lambda *_: None):
    """
    Fine-tunes in place (Adam, early stopping on val accuracy, best weights restored).
    Modes: fc (FC trainable, BN running stats adapt – run01's mode), fc_frozen (strict T1), full (T4).
    Returns (best record, history).
    """
    params, bn_eval = configure(model, mode)
    opt = torch.optim.Adam(params, lr=lr)
    try:
        return fit(model, train_loader, val_loader, opt, max_epochs, patience,
                   monitor="val_acc", log=log, bn_eval=bn_eval)
    finally:
        for p in model.parameters():
            p.requires_grad = True
        model.eval()


# ── features before the last FC layer ─────────────────────────

class _FcInput:
    """Context manager: records the input of the last FC layer on every forward."""

    def __init__(self, model):
        self.fc, self.x = _last_fc(model), None

    def __enter__(self):
        self.h = self.fc.register_forward_pre_hook(lambda _m, inp: setattr(self, "x", inp[0].detach()))
        return self

    def __exit__(self, *exc):
        self.h.remove()


@torch.no_grad()
def collect_features(model, loader, max_batches=None):
    """(features (N, D) float64 CPU, labels (N,) CPU) at the input of the last FC layer, eval mode."""
    model.eval()
    feats, ys = [], []
    with _FcInput(model) as rec:
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            model(x)
            feats.append(rec.x.flatten(1).double().cpu()); ys.append(y.cpu())
    return torch.cat(feats), torch.cat(ys)


# ── T1: closed-form ridge refit of the head ───────────────────

def _targets(kind, y, x, teacher, C):
    if kind == "logits":
        return teacher(x)
    if kind == "labels":
        return F.one_hot(y, C).to(x.dtype) - 1.0 / C
    raise ValueError(kind)


@torch.no_grad()
def ridge_head(model, train_loader, val_loader, lambdas, target="logits", teacher=None, max_batches=None):
    """
    Refit the last FC layer (W, b) in closed form, in place:
        min_W,b  Σ ||W x_n + b − t_n||²  +  λ ||W||²_F        (bias not penalised)
    x_n = features at the FC input on `train_loader`, t_n = teacher logits (target="logits", teacher
    required, e.g. the dense model) or centred one-hot labels (target="labels").

    Solver. The normal equations are accumulated batch by batch, so the N × D feature matrix is never stored:
    G = Σ x xᵀ (D × D), B = Σ x tᵀ (D × C), plus the sums for centring. Then Gc = G − N μμᵀ is
    eigendecomposed once (Gc = Q diag(s) Qᵀ), and every λ costs only a D × D × C product:
    W_λ = Q diag(1/(s+λ)) Qᵀ Bc. λ is applied relative to mean(s), so the grid does not depend on the
    feature scale. Precision is float64 for D ≤ 8192 and float32 above that (Conv2Net dense: D = 16384,
    so G is 1 GiB in float32 instead of 2 GiB in float64). On the device of `model`.

    λ is chosen by val accuracy (val labels). Returns dict(lambda, val_acc, per_lambda {λ: val_acc}).
    """
    if target == "logits" and teacher is None:
        raise ValueError("target='logits' needs the teacher (dense) model")
    fc = _last_fc(model)
    dev, C, D = _device(model), fc.out_features, fc.in_features
    dt = torch.float64 if D <= 8192 else torch.float32
    model.eval()
    if teacher is not None:
        teacher.eval()

    G = torch.zeros(D, D, device=dev, dtype=dt)
    B = torch.zeros(D, C, device=dev, dtype=dt)
    sx = torch.zeros(D, device=dev, dtype=torch.float64)
    st = torch.zeros(C, device=dev, dtype=torch.float64)
    n = 0
    with _FcInput(model) as rec:
        for i, (x, y) in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break
            model(x)
            f = rec.x.flatten(1)
            t = _targets(target, y, x, teacher, C)
            G.addmm_(f.T.to(dt), f.to(dt)); B.addmm_(f.T.to(dt), t.to(dt))
            sx += f.double().sum(0); st += t.double().sum(0); n += len(y)

    mu, tbar = sx / n, st / n
    Gc = G - n * torch.outer(mu, mu).to(dt)
    Bc = B - n * torch.outer(mu, tbar).to(dt)
    del G, B
    s, Q = torch.linalg.eigh(Gc)
    del Gc
    s = s.clamp_min(0)
    QtB = Q.T @ Bc
    scale = float(s.mean())

    Ws, bs = [], []
    for lam in lambdas:
        W = Q @ (QtB / (s + lam * scale)[:, None])              # (D, C)
        Ws.append(W.double())
        bs.append(tbar - mu @ W.double())
    del Q

    # λ by val accuracy, all λ evaluated in one pass
    correct = torch.zeros(len(lambdas), dtype=torch.long)
    m = 0
    with _FcInput(model) as rec:
        for x, y in val_loader:
            model(x)
            f = rec.x.flatten(1).double()
            for j, (W, b) in enumerate(zip(Ws, bs)):
                correct[j] += ((f @ W + b).argmax(1) == y).sum().cpu()
            m += len(y)
    accs = (100.0 * correct.double() / m).tolist()
    j = max(range(len(lambdas)), key=lambda k: (accs[k], lambdas[k]))   # ties → larger λ (more stable)
    fc.weight.copy_(Ws[j].T.to(fc.weight.dtype))
    if fc.bias is not None:
        fc.bias.copy_(bs[j].to(fc.bias.dtype))
    else:
        raise ValueError("ridge_head expects a last FC layer with bias")
    return dict(**{"lambda": lambdas[j]}, val_acc=accs[j], per_lambda=dict(zip(lambdas, accs)))


# ── T2: BatchNorm recalibration ───────────────────────────────

@torch.no_grad()
def bn_recalibrate(model, loader, n_batches):
    """
    Reset every BatchNorm's running statistics and re-estimate them as the cumulative average
    (momentum=None) over `n_batches` batches of `loader`. Only BN modules are put in train mode
    (dropout stays off). Original momenta are restored; the model is left in eval mode.
    """
    bns = _bns(model)
    momenta = [b.momentum for b in bns]
    model.eval()
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
        b.train()
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x)
    for b, mom in zip(bns, momenta):
        b.momentum = mom
    model.eval()
    return model


# ── T3: least-squares compensation of the next conv ───────────

@torch.no_grad()
def ls_compensate(dense, pruned, keep, loader, n_batches, ridge=1e-3, positions=64, seed=0):
    """
    For each (conv, bn, next) in CONV_SEQ with a next conv, refit the pruned next conv's weights and bias so
    that, on the pruned network's own input to it (kept channels only), its pre-BN output reproduces the
    DENSE next conv's pre-BN output on the dense network (restricted to the next conv's kept filters):

        min_w  Σ ||w · z − t||² + ridge · mean(diag A) · ||w_weights||²,   z = [im2col patch ; 1]

    Layers are processed in forward order, each with a fresh pass, so later layers see the already
    compensated earlier ones (sequential, as in channel-pruning / ThiNet). `positions` random spatial
    positions per image bound memory. The last conv's removed channels are not handled here (the head
    sees them; use ridge_head).

    Returns {next conv name: dict(before=rel_err, after=rel_err)}, relative squared error on the samples.
    """
    dense.eval(); pruned.eval()
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for _conv, _bn, nxt in pruned.CONV_SEQ:
        if nxt is None:
            continue
        pc, dc = getattr(pruned, nxt), getattr(dense, nxt)
        dev = pc.weight.device
        k_out = torch.as_tensor(keep[nxt]).nonzero().flatten().to(dev) if nxt in keep else \
            torch.arange(pc.out_channels, device=dev)
        rec = {}
        h1 = pc.register_forward_pre_hook(lambda _m, inp: rec.__setitem__("z", inp[0]))
        h2 = dc.register_forward_hook(lambda _m, _i, o: rec.__setitem__("t", o))
        P = pc.in_channels * pc.kernel_size[0] * pc.kernel_size[1] + 1
        A = torch.zeros(P, P, dtype=torch.float64, device=dev)
        Bm = torch.zeros(P, len(k_out), dtype=torch.float64, device=dev)
        tt = 0.0
        try:
            for i, (x, _) in enumerate(loader):
                if i >= n_batches:
                    break
                pruned(x); dense(x)
                cols = F.unfold(rec["z"], pc.kernel_size, pc.dilation, pc.padding, pc.stride)  # (b, P-1, L)
                t = rec["t"][:, k_out].flatten(2)                    # (b, out, L), dense pre-BN incl. bias
                L = cols.shape[-1]
                idx = torch.randint(0, L, (min(positions, L),), generator=gen).to(dev)
                z = cols[..., idx].permute(0, 2, 1).reshape(-1, P - 1).double()
                z = torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype, device=dev)], 1)
                tg = t[..., idx].permute(0, 2, 1).reshape(-1, len(k_out)).double()
                A += z.T @ z; Bm += z.T @ tg; tt += float((tg ** 2).sum())
        finally:
            h1.remove(); h2.remove()

        def rel_err(w):
            return float((torch.sum(w * (A @ w)) - 2 * torch.sum(w * Bm) + tt) / max(tt, 1e-12))

        w_old = torch.cat([pc.weight.reshape(pc.out_channels, -1).double(),
                           (pc.bias.double() if pc.bias is not None
                            else torch.zeros(pc.out_channels, dtype=torch.float64, device=dev))[:, None]], 1).T
        reg = torch.full((P,), ridge * float(torch.diagonal(A)[:-1].mean()), dtype=torch.float64, device=dev)
        reg[-1] = 0.0
        w_new = torch.linalg.solve(A + torch.diag(reg), Bm)                                   # (P, out)
        before, after = rel_err(w_old), rel_err(w_new)
        pc.weight.copy_(w_new[:-1].T.reshape_as(pc.weight).to(pc.weight.dtype))
        if pc.bias is not None:
            pc.bias.copy_(w_new[-1].to(pc.bias.dtype))
        out[nxt] = dict(before=before, after=after)
    return out
