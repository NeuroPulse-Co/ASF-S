"""
Step 8 tests: what each fine-tuning mode trains, and what it leaves untouched.
"""

import numpy as np
import pytest
import torch

import data
import models
from asfs.finetune import configure, finetune


def setup(seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    split = data.Split(rng.integers(0, 256, (64, 32, 32, 3), dtype=np.uint8),
                       rng.integers(0, 10, 64).astype(np.int64), "tiny")
    dl = data.loader(split, 32, False, torch.device("cpu"), "gpu")
    return models.build("conv2net"), dl


def snapshot(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


@pytest.mark.parametrize("mode, conv_changes, bn_stats_change", [
    ("fc", False, True),             # original code: convs frozen, BN running stats still adapt
    ("fc_frozen", False, False),     # strict freeze
    ("full", True, True),
])
def test_modes(mode, conv_changes, bn_stats_change):
    m, dl = setup()
    before = snapshot(m)
    finetune(m, mode, dl, dl, lr=1e-2, max_epochs=2, patience=10)
    after = m.state_dict()
    changed = lambda k: not torch.equal(before[k], after[k])
    assert changed("fc.weight")
    assert changed("conv1.weight") == conv_changes and changed("bn1.weight") == conv_changes
    assert changed("bn1.running_mean") == bn_stats_change


def test_configure_trainable_sets():
    m, _ = setup()
    p, bn_eval = configure(m, "fc")
    assert {id(x) for x in p} == {id(x) for x in m.fc.parameters()} and not bn_eval
    assert not m.conv1.weight.requires_grad
    _, bn_eval = configure(m, "fc_frozen")
    assert bn_eval
    p, _ = configure(m, "full")
    assert len(p) == len(list(m.parameters())) and all(x.requires_grad for x in p)
