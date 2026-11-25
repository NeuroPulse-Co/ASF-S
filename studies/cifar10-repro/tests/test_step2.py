"""
Step 2 tests: extraction returns the output layer in dataset order, and the
correct/incorrect grouping is right. CPU, synthetic data.
"""

import numpy as np
import torch

import data
import models
from step2_activations import extract, group_counts


def test_extract_matches_forward_in_order():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    split = data.Split(rng.integers(0, 256, (70, 32, 32, 3), dtype=np.uint8),
                       rng.integers(0, 10, 70).astype(np.int64), "tiny")
    model = models.build("conv2net").eval()
    acts, labels, preds = extract(model, data.loader(split, 32, False, torch.device("cpu"), "gpu"))
    x = data.normalize(torch.from_numpy(split.images).permute(0, 3, 1, 2).float() / 255)
    with torch.no_grad():
        ref = model(x).numpy()
    assert acts.shape == (70, 10)
    np.testing.assert_allclose(acts, ref, atol=1e-5)
    assert (labels == split.labels).all() and (preds == ref.argmax(1)).all()


def test_group_counts():
    labels = np.array([0, 0, 1, 1, 1, 2])
    correct = np.array([True, False, True, True, False, False])
    c = group_counts(labels, correct, num_classes=3)
    assert c.tolist() == [[1, 1], [2, 1], [0, 1]]
