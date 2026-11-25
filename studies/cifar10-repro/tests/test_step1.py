"""
Step 1 tests: the training loop learns, early stopping fires, best weights are restored.
CPU, synthetic data, no download.
"""

import numpy as np
import torch

import data
import models
from asfs.train import evaluate, fit


def tiny_split(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return data.Split(rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8),
                      rng.integers(0, 10, n).astype(np.int64), "tiny")


def test_fit_overfits_and_restores_best():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    split = tiny_split()
    tr = data.loader(split, 32, False, dev, backend="gpu")    # no augmentation: memorise
    model = models.build("conv2net")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _, acc0 = evaluate(model, tr)
    best, hist = fit(model, tr, tr, opt, max_epochs=30, patience=50, monitor="val_loss",
                     log=lambda *_: None)
    loss, acc = evaluate(model, tr)
    assert acc > acc0 + 50, (acc0, acc)
    assert abs(loss - best["val_loss"]) < 1e-5            # best weights restored


def test_early_stop():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    split = tiny_split(32)
    dl = data.loader(split, 32, False, dev, backend="gpu")
    model = models.build("conv2net")
    for m in model.modules():                              # freeze BN running stats too
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 0.0
    opt = torch.optim.SGD(model.parameters(), lr=0.0)       # nothing changes → no improvement
    _, hist = fit(model, dl, dl, opt, max_epochs=20, patience=3, log=lambda *_: None)
    assert len(hist) == 4                                  # 1 best + 3 without improvement
