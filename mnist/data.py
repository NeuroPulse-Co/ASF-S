"""
data.py - MNIST as in-memory tensors, and the analysis / eval split of the test set.

Data roles in this experiment
-----------------------------
    MNIST train (60k)       -> fine-tuning after pruning (the checkpoint was trained on it)
    MNIST test, "analysis"  -> (a) building PG1: we need MISCLASSIFIED samples, and the
                                   checkpoint is 99.97% accurate on train, so train has
                                   ~17 errors in total (only 1 in the first 1000/class the
                                   original script used; see paper Fig. 14's empty panels).
                               (b) early stopping / model selection during fine-tuning.
    MNIST test, "eval"      -> ONLY for reporting final accuracy / ECE.

Keeping "eval" untouched by every decision (which neurons to prune, when to stop
fine-tuning) is what makes the reported numbers an honest estimate.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

import config


def _to_tensors(ds):
    """uint8 images (N, 28, 28) -> normalised float32 (N, 1, 28, 28), labels int64.

    Doing the normalisation once, up front, is equivalent to
    transforms.Compose([ToTensor(), Normalize(mean, std)]) applied per image, but it
    avoids re-decoding every image on every epoch. MNIST as float32 is ~190 MB."""
    x = ds.data.float().div_(255.0).sub_(config.MNIST_MEAN).div_(config.MNIST_STD)
    return x.unsqueeze(1), ds.targets.clone()


def load_mnist():
    """Returns (x_train, y_train, x_test, y_test). Downloads MNIST on first use (~11 MB)."""
    tr = datasets.MNIST(config.DATA_DIR, train=True, download=True)
    te = datasets.MNIST(config.DATA_DIR, train=False, download=True)
    return (*_to_tensors(tr), *_to_tensors(te))


def split_test(y_test, seed=config.SPLIT_SEED):
    """Stratified 50/50 split of the test indices: every class is halved separately,
    so both halves have the same class balance. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    analysis, evaluation = [], []
    y = y_test.numpy()
    for c in range(config.NUM_CLASSES):
        idx = rng.permutation(np.where(y == c)[0])
        half = len(idx) // 2
        analysis.append(idx[:half])
        evaluation.append(idx[half:])
    return np.sort(np.concatenate(analysis)), np.sort(np.concatenate(evaluation))


def get_data():
    """One call that every stage uses, so they all see identical splits.

    Returns a dict of tensors:
        train_x/train_y, analysis_x/analysis_y, eval_x/eval_y, plus the index arrays
        analysis_idx / eval_idx (positions within the official MNIST test set)."""
    x_tr, y_tr, x_te, y_te = load_mnist()
    a_idx, e_idx = split_test(y_te)
    return {
        "train_x": x_tr, "train_y": y_tr,
        "analysis_x": x_te[a_idx], "analysis_y": y_te[a_idx], "analysis_idx": a_idx,
        "eval_x": x_te[e_idx], "eval_y": y_te[e_idx], "eval_idx": e_idx,
    }


def make_loader(x, y, batch_size, shuffle, seed=config.SEED):
    """DataLoader over in-memory tensors. num_workers=0 is deliberate: with data already
    in RAM, worker processes only add start-up cost (and are fragile on Windows)."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, generator=g)
