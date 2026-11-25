"""
train.py – one-epoch training, evaluation, and the early-stopping loop.

Shared by step 1 (training the dense model) and step 8 (fine-tuning pruned models).
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_train_mode(model, bn_eval=False):
    """model.train(), optionally keeping every BatchNorm in eval mode (running stats frozen)."""
    model.train()
    if bn_eval:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def train_epoch(model, loader, opt, bn_eval=False):
    set_train_mode(model, bn_eval)
    total_loss = correct = n = 0
    for x, y in loader:
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, 100.0 * correct / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = correct = n = 0
    for x, y in loader:
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, 100.0 * correct / n


def fit(model, train_loader, val_loader, opt, max_epochs, patience,
        monitor="val_loss", scheduler=None, log=print, bn_eval=False):
    """
    Train until `monitor` stops improving for `patience` epochs or max_epochs is reached.
    monitor: "val_loss" (lower is better) or "val_acc" (higher is better).
    scheduler: ReduceLROnPlateau, stepped on val loss each epoch.
    bn_eval: keep BatchNorm in eval mode while training (no running-stat updates).
    Restores the best weights. Returns (best_record, history).
    """
    sign = 1 if monitor == "val_loss" else -1
    best_score, best_state, best = float("inf"), None, None
    history, bad = [], 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, bn_eval)
        va_loss, va_acc = evaluate(model, val_loader)
        if scheduler is not None:
            scheduler.step(va_loss)
        rec = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                   val_acc=va_acc, lr=opt.param_groups[0]["lr"], seconds=time.time() - t0)
        history.append(rec)
        score = sign * rec[monitor]
        improved = score < best_score
        if improved:
            best_score, best, bad = score, rec, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        log(f"  ep {epoch:3d}  train {tr_loss:.4f}/{tr_acc:5.2f}  val {va_loss:.4f}/{va_acc:5.2f}"
            f"  lr {rec['lr']:.1e}  {rec['seconds']:.1f}s{'  *' if improved else ''}")
        if bad >= patience:
            log(f"  early stop: no {monitor} improvement for {patience} epochs")
            break
    model.load_state_dict(best_state)
    return best, history
