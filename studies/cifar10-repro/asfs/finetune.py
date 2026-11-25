"""
finetune.py – recovering accuracy after pruning [paper §5.3–5.4, ASF-S (a) and (b)].

Modes:
  fc          only the FC layers are trainable; BatchNorm stays in train mode, so its running
              statistics re-adapt to the pruned network. This is what the original code does
              (cifar100 fine_tune_fc calls model.train() with the convs frozen). Paper ASF-S(b).
  fc_frozen   only the FC layers are trainable and BatchNorm is in eval mode: the pruned layers
              are frozen completely, the strict reading of "freezing the pruned layers".
  full        every layer trainable. Paper ASF-S(a).

Recipe [assumed: original pg_pruning_conv2.py]: Adam lr 1e-4, augmented train set, early
stopping on val accuracy (patience 10, ≤ 200 epochs), best weights restored.
"""

import torch

from asfs.train import fit

MODES = ("fc", "fc_frozen", "full")


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
    """Fine-tunes in place. Returns (best record, history)."""
    params, bn_eval = configure(model, mode)
    opt = torch.optim.Adam(params, lr=lr)
    return fit(model, train_loader, val_loader, opt, max_epochs, patience,
               monitor="val_acc", log=log, bn_eval=bn_eval)
