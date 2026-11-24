"""
make_golden.py – freeze the reference outputs of the original cifar10/asfs code (plan P13).

Runs cifar10/asfs/{afc,pg,contrast,pgi,prune} on fixed synthetic inputs (a seeded Conv2Net/Conv6Net with
non-trivial BN statistics, seeded "activations") and saves everything to golden.npz. tests/test_golden.py
compares afcprune against this file, so the comparison keeps working after afcprune leaves this repo.

Regenerate (only when the reference code changes, from the repo root, which must contain cifar10/):
    afcprune/.venv/bin/python afcprune/tests/golden/make_golden.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "cifar10"))

import models                                  # noqa: E402  (cifar10/models.py)
from asfs import afc, contrast, pg, pgi, prune  # noqa: E402

SIGMA, EPS, ALPHA, K = 0.8, 1e-6, 0.5, 4


def seeded_model(name, seed=0):
    torch.manual_seed(seed)
    m = models.build(name)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.5, 0.5)
            mod.running_mean.uniform_(-0.2, 0.2); mod.running_var.uniform_(0.5, 2.0)
    return m.eval()


def main():
    rng = np.random.default_rng(0)
    out = {}
    # AFC + PG1 on synthetic output-layer activations: 10 classes × {correct, incorrect}, S = 50, N = 10
    acts = rng.normal(size=(10, 2, 50, 10)) + rng.normal(size=(10, 2, 1, 10))
    out["acts"] = acts
    g = np.zeros((10, 2, 10)); lam = np.zeros((10, 2, 10)); W_all = np.zeros((10, 2, 10, 10))
    for c in range(10):
        for s in range(2):
            A, W = afc.afc(acts[c, s], SIGMA, EPS)
            W_all[c, s] = W
            g[c, s], lam[c, s] = pg.pg1(W, ALPHA, "paper")
    out["W"], out["g"], out["lam"] = W_all, g, lam
    out["median_sigma"] = afc.median_sigma([afc.cosine_similarity(afc.activation_matrix(acts[c, s]))
                                            for c in range(10) for s in range(2)])
    dG, flip = contrast.contrast_matrix(g[:, 0], g[:, 1])
    comps, ratio, _ = contrast.pca(dG, K)
    out["dG"], out["flip"], out["comps"], out["ratio"] = dG, flip, comps, ratio

    x = torch.from_numpy(rng.normal(size=(4, 3, 32, 32)).astype(np.float32))
    out["x"] = x.numpy()
    for name in ("conv2net", "conv6net"):
        m = seeded_model(name)
        for mode in ("aligned", "literal"):
            for layer, v in pgi.pgi(m, comps, mode).items():
                out[f"pgi_{mode}__{name}__{layer}"] = v
        keep = {c: rng.random(getattr(m, c).out_channels) < 0.6 for c, _, _ in m.CONV_SEQ}
        for c, v in keep.items():
            out[f"keep__{name}__{c}"] = v
        with torch.no_grad():
            out[f"logits_pruned__{name}"] = prune.prune(m, keep)(x).numpy()
    np.savez_compressed(HERE / "golden.npz", **out)
    print(f"wrote {HERE / 'golden.npz'} ({len(out)} arrays)")


if __name__ == "__main__":
    main()
