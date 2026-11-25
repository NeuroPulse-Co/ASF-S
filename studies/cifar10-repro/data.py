"""
data.py – CIFAR-10 splits for the two protocols, and batch iterators.

Two interchangeable backends yield normalised (x, y) batches on the device:

  gpu          the whole split lives on the GPU as uint8; augmentation runs per sample,
               in batches (gpu_augment). Removes the CPU data-loading bottleneck that
               held the old pipeline at ~23 s/epoch.
  torchvision  the original scripts' PIL transforms through a DataLoader. Kept as the
               reference the gpu backend is tested against.

Augmentation follows paper App. C in the original order:
  RandomHorizontalFlip → RandomCrop(32, pad 4) → ColorJitter(0.2,0.2,0.2,0.1) → RandomRotation(15)
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import config as C

_MEAN = torch.tensor(C.MEAN).view(1, 3, 1, 1)
_STD  = torch.tensor(C.STD).view(1, 3, 1, 1)


# ── Splits ────────────────────────────────────────────────────

class Split:
    """A labelled image set held as uint8 NHWC numpy + int64 labels."""

    def __init__(self, images, labels, name):
        self.images = images            # (N, 32, 32, 3) uint8
        self.labels = labels            # (N,) int64
        self.name   = name

    def __len__(self):
        return len(self.labels)

    def subset(self, idx, name):
        return Split(self.images[idx], self.labels[idx], name)


def _load(train):
    ds = datasets.CIFAR10(C.DATA_DIR, train=train, download=True)
    return Split(np.asarray(ds.data), np.asarray(ds.targets, dtype=np.int64),
                 "train" if train else "test")


def stratified_split(labels, n_val, seed=C.SEED):
    """Indices (train_idx, val_idx) with n_val / num_classes samples of each class in val."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = n_val // len(classes)
    val = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:per_class]
                          for c in classes])
    train = np.setdiff1d(np.arange(len(labels)), val)
    return np.sort(train), np.sort(val)


def n_per_class(labels, n, seed=C.SEED):
    """Sorted indices of the first n samples of each class after a seeded shuffle."""
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:n]
                                   for c in np.unique(labels)]))


def get_splits(protocol=C.PROTOCOL):
    """
    Returns dict with:
      train – fit the model on this
      val   – early stopping, fine-tune checkpoint and τ selection
      test  – final numbers
      pg    – images whose output-layer activations build PG1 (step 2)
    Under the paper protocol val *is* test, as in the original scripts.
    """
    full, test = _load(True), _load(False)
    if protocol == "paper":
        pg = full.subset(n_per_class(full.labels, C.ACT_PER_CLASS), "train-subset")
        return dict(train=full, val=test, test=test, pg=pg)
    if protocol == "clean":
        tr, va = stratified_split(full.labels, C.VAL_SIZE)
        val = full.subset(va, "val")
        return dict(train=full.subset(tr, "train45k"), val=val, test=test, pg=val)
    raise ValueError(f"unknown protocol {protocol!r}")


# ── GPU augmentation ──────────────────────────────────────────

def _gray(x):
    return (0.2989 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def _blend(a, b, ratio):
    return (ratio * a + (1 - ratio) * b).clamp(0, 1)


def _rgb2hsv(x):
    r, g, b = x.unbind(1)
    maxc, _ = x.max(1)
    minc, _ = x.min(1)
    eqc  = maxc == minc
    cr   = maxc - minc
    ones = torch.ones_like(maxc)
    s    = cr / torch.where(eqc, ones, maxc)
    crd  = torch.where(eqc, ones, cr)
    rc, gc, bc = (maxc - r) / crd, (maxc - g) / crd, (maxc - b) / crd
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h  = torch.fmod((hr + hg + hb) / 6.0 + 1.0, 1.0)
    return torch.stack((h, s, maxc), 1)


def _hsv2rgb(x):
    h, s, v = x.unbind(1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.to(torch.int64) % 6
    p = (v * (1 - s)).clamp(0, 1)
    q = (v * (1 - s * f)).clamp(0, 1)
    t = (v * (1 - s * (1 - f))).clamp(0, 1)
    mask = i.unsqueeze(1) == torch.arange(6, device=x.device).view(1, -1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), 1)
    a2 = torch.stack((t, v, v, q, p, p), 1)
    a3 = torch.stack((p, p, t, v, v, q), 1)
    return torch.stack([(mask * a).sum(1) for a in (a1, a2, a3)], 1)


def _uniform(n, lo, hi, device, gen):
    return torch.rand(n, device=device, generator=gen) * (hi - lo) + lo


# Per-sample ops. Each takes x (B, 3, H, W) in [0, 1] and one parameter per sample, and
# matches the torchvision.transforms.functional op of the same name (tests/test_data.py).

def hflip(x, flip):
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)


def crop(x, oy, ox, pad=C.CROP_PAD):
    """RandomCrop(H, padding=pad) with zero fill; (oy, ox) is the top-left corner in the padded image."""
    B, _, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    rows = (oy.view(B, 1) + torch.arange(H, device=x.device)).view(B, 1, H, 1)
    cols = (ox.view(B, 1) + torch.arange(W, device=x.device)).view(B, 1, 1, W)
    bi = torch.arange(B, device=x.device).view(B, 1, 1, 1)
    ci = torch.arange(x.shape[1], device=x.device).view(1, -1, 1, 1)
    return xp[bi, ci, rows, cols]


def adjust_brightness(x, f):
    return (x * f.view(-1, 1, 1, 1)).clamp(0, 1)


def adjust_contrast(x, f):
    return _blend(x, _gray(x).mean((1, 2, 3), keepdim=True), f.view(-1, 1, 1, 1))


def adjust_saturation(x, f):
    return _blend(x, _gray(x), f.view(-1, 1, 1, 1))


def adjust_hue(x, f):
    hsv = _rgb2hsv(x)
    return _hsv2rgb(torch.stack(((hsv[:, 0] + f.view(-1, 1, 1)) % 1.0, hsv[:, 1], hsv[:, 2]), 1))


def rotate(x, deg):
    """Counter-clockwise rotation about the centre, nearest, zero fill (as RandomRotation)."""
    a = deg * math.pi / 180
    cos, sin, zero = torch.cos(a), torch.sin(a), torch.zeros_like(a)
    theta = torch.stack((torch.stack((cos, -sin, zero), 1),
                         torch.stack((sin,  cos, zero), 1)), 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def gpu_augment(x, gen=None):
    """
    x: (B, 3, 32, 32) float in [0, 1]. Returns the augmented batch, same shape.
    Every random parameter is drawn per sample, including ColorJitter's random order of
    its four ops (as torchvision draws a new order for every image).
    """
    B, dev, j = x.shape[0], x.device, C.JITTER
    x = hflip(x, torch.rand(B, device=dev, generator=gen) < 0.5)
    x = crop(x, *(torch.randint(0, 2 * C.CROP_PAD + 1, (B,), device=dev, generator=gen)
                  for _ in range(2)))
    factors = [_uniform(B, 1 - j["brightness"], 1 + j["brightness"], dev, gen),
               _uniform(B, 1 - j["contrast"], 1 + j["contrast"], dev, gen),
               _uniform(B, 1 - j["saturation"], 1 + j["saturation"], dev, gen),
               _uniform(B, -j["hue"], j["hue"], dev, gen)]
    order = torch.rand(B, 4, device=dev, generator=gen).argsort(1)
    x = color_jitter(x, factors, order)
    return rotate(x, _uniform(B, -C.ROTATION, C.ROTATION, dev, gen))


JITTER_OPS = [adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue]


def color_jitter(x, factors, order):
    """
    factors: 4 tensors (B,) for brightness, contrast, saturation, hue.
    order:   (B, 4) long; row b is the op sequence for sample b (a permutation of 0..3).
    """
    B = x.shape[0]
    for r in range(4):
        out = x
        for k, (op, f) in enumerate(zip(JITTER_OPS, factors)):
            sel = order[:, r] == k
            if sel.any():
                out = torch.where(sel.view(B, 1, 1, 1), op(x, f), out)
        x = out
    return x


def normalize(x):
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


# ── Batch iterators ───────────────────────────────────────────

class GpuLoader:
    """Iterates a Split held on the device. Yields normalised (x, y)."""

    def __init__(self, split, batch_size, train, device, seed=C.SEED):
        self.x = torch.from_numpy(split.images).to(device).permute(0, 3, 1, 2).contiguous()
        self.y = torch.from_numpy(split.labels).to(device)
        self.bs, self.train, self.device = batch_size, train, device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def __len__(self):
        return math.ceil(len(self.y) / self.bs)

    def __iter__(self):
        n = len(self.y)
        order = (torch.randperm(n, device=self.device, generator=self.gen) if self.train
                 else torch.arange(n, device=self.device))
        for i in range(0, n, self.bs):
            idx = order[i:i + self.bs]
            x = self.x[idx].float().div_(255)
            if self.train:
                x = gpu_augment(x, self.gen)
            yield normalize(x), self.y[idx]


class _NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, split, tf):
        self.split, self.tf = split, tf

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.fromarray(self.split.images[i])), int(self.split.labels[i])


def torchvision_transforms(train):
    tf = [transforms.RandomHorizontalFlip(),
          transforms.RandomCrop(32, padding=C.CROP_PAD),
          transforms.ColorJitter(**C.JITTER),
          transforms.RandomRotation(C.ROTATION)] if train else []
    return transforms.Compose(tf + [transforms.ToTensor(), transforms.Normalize(C.MEAN, C.STD)])


class TorchvisionLoader:
    """The original scripts' pipeline; yields normalised (x, y) on the device."""

    def __init__(self, split, batch_size, train, device, seed=C.SEED):
        g = torch.Generator().manual_seed(seed)
        self.dl = torch.utils.data.DataLoader(
            _NumpyDataset(split, torchvision_transforms(train)), batch_size=batch_size,
            shuffle=train, generator=g, num_workers=C.NUM_WORKERS,
            pin_memory=device.type == "cuda", persistent_workers=C.NUM_WORKERS > 0)
        self.device = device

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for x, y in self.dl:
            yield x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def loader(split, batch_size, train, device, backend=C.AUGMENT_BACKEND, seed=C.SEED):
    cls = {"gpu": GpuLoader, "torchvision": TorchvisionLoader}[backend]
    return cls(split, batch_size, train, device, seed)
