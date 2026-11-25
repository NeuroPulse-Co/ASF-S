"""
Step 0 tests: architecture sizes, protocol splits, and the GPU augmentation against
torchvision's own functional ops. CPU only, no dataset download.
"""

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import data
import models

torch.manual_seed(0)


# ── Architecture [paper Table 4] ──────────────────────────────

@pytest.mark.parametrize("name, n", [("conv2net", 183_434), ("conv6net", 815_018)])
def test_param_count(name, n):
    m = models.build(name)
    assert sum(p.numel() for p in m.parameters()) == n


@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
def test_forward_shape(name):
    assert models.build(name)(torch.randn(2, 3, 32, 32)).shape == (2, 10)


# ── Splits ────────────────────────────────────────────────────

def test_stratified_split():
    labels = np.repeat(np.arange(10), 5000)
    tr, va = data.stratified_split(labels, 5000)
    assert len(va) == 5000 and len(tr) == 45000
    assert len(np.intersect1d(tr, va)) == 0
    assert (np.bincount(labels[va]) == 500).all()
    tr2, va2 = data.stratified_split(labels, 5000)
    assert (va == va2).all()                                   # seeded


def test_n_per_class():
    labels = np.repeat(np.arange(10), 5000)
    idx = data.n_per_class(labels, 1000)
    assert (np.bincount(labels[idx]) == 1000).all()


# ── GPU augmentation vs torchvision ───────────────────────────

@pytest.fixture
def img():
    return torch.rand(4, 3, 32, 32)


def test_hflip(img):
    flip = torch.tensor([True, False, True, False])
    out = data.hflip(img, flip)
    for i in range(4):
        ref = TF.hflip(img[i]) if flip[i] else img[i]
        assert torch.equal(out[i], ref)


def test_crop(img):
    oy, ox = torch.tensor([0, 4, 8, 3]), torch.tensor([8, 4, 0, 5])
    out = data.crop(img, oy, ox)
    for i in range(4):
        ref = TF.crop(TF.pad(img[i], 4), int(oy[i]), int(ox[i]), 32, 32)
        assert torch.equal(out[i], ref)


@pytest.mark.parametrize("ours, theirs, factors", [
    (data.adjust_brightness, TF.adjust_brightness, [0.8, 1.0, 1.1, 1.2]),
    (data.adjust_contrast,   TF.adjust_contrast,   [0.8, 1.0, 1.1, 1.2]),
    (data.adjust_saturation, TF.adjust_saturation, [0.8, 1.0, 1.1, 1.2]),
    (data.adjust_hue,        TF.adjust_hue,        [-0.1, 0.0, 0.05, 0.1]),
])
def test_color_ops(img, ours, theirs, factors):
    out = ours(img, torch.tensor(factors))
    for i, f in enumerate(factors):
        torch.testing.assert_close(out[i], theirs(img[i], f), atol=1e-5, rtol=0)


def test_color_jitter_per_sample_order(img):
    factors = [torch.tensor([0.8, 1.2, 0.9, 1.1]), torch.tensor([1.2, 0.8, 1.1, 0.9]),
               torch.tensor([0.8, 1.2, 1.2, 0.8]), torch.tensor([0.1, -0.1, 0.05, -0.05])]
    order = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]])
    tv = [TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation, TF.adjust_hue]
    out = data.color_jitter(img, factors, order)
    for b in range(4):
        ref = img[b]
        for k in order[b].tolist():
            ref = tv[k](ref, float(factors[k][b]))
        torch.testing.assert_close(out[b], ref, atol=1e-5, rtol=0)


def test_hsv_roundtrip(img):
    torch.testing.assert_close(data._hsv2rgb(data._rgb2hsv(img)), img, atol=1e-5, rtol=0)


@pytest.mark.parametrize("deg", [-15.0, -7.0, 0.0, 9.0, 15.0])
def test_rotate(img, deg):
    out = data.rotate(img[:1], torch.tensor([deg]))[0]
    ref = TF.rotate(img[0], deg, interpolation=InterpolationMode.NEAREST)
    # nearest-neighbour rounding can differ on a few boundary pixels
    agree = (out - ref).abs().max(0).values.lt(1e-6).float().mean().item()
    assert agree > 0.97, f"only {agree:.3f} of pixels match torchvision for {deg}°"


def test_gpu_augment_properties():
    x = torch.rand(1, 3, 32, 32).repeat(64, 1, 1, 1)
    g = torch.Generator().manual_seed(0)
    out = data.gpu_augment(x, g)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    # parameters are per sample: identical inputs give (almost) all distinct outputs
    flat = out.flatten(1)
    assert torch.cdist(flat, flat).add(torch.eye(64) * 1e9).min() > 1e-3
    # seeded → reproducible
    torch.testing.assert_close(data.gpu_augment(x, torch.Generator().manual_seed(0)), out)
