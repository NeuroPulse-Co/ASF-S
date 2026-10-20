"""
diag_aug_parity – do the gpu and torchvision augmentations produce the same distribution?

Each backend augments the same N training images R times, with no training involved.
Per-image statistics of the outputs are then compared with two-sample KS tests:
  mean      mean pixel value                      (brightness)
  std       pixel std over the image              (contrast)
  sat       mean of max−min over RGB per pixel    (saturation)
  zero      fraction of pixels that are 0 in all channels (crop padding + rotation fill)
  dist      mean |augmented − original|           (overall strength)
  hue_shift mean hue difference vs original       (hue)

A real difference shows up as a large KS statistic (> ~0.05 with these sample sizes)
in one of the stats, which also points to the op responsible.
"""

import json

import numpy as np
import torch
from PIL import Image
from scipy.stats import ks_2samp
from torchvision import transforms

from common import C, parse_args, run_dir, seed_everything
import data


def stats(aug, orig):
    """aug, orig: (N, 3, H, W) float in [0, 1]. Returns dict of (N,) numpy arrays."""
    hsv_a, hsv_o = data._rgb2hsv(aug), data._rgb2hsv(orig)
    dh = (hsv_a[:, 0] - hsv_o[:, 0] + 0.5) % 1.0 - 0.5
    return {
        "mean": aug.mean((1, 2, 3)),
        "std": aug.std((1, 2, 3)),
        "sat": (aug.max(1).values - aug.min(1).values).mean((1, 2)),
        "zero": (aug.amax(1) == 0).float().mean((1, 2)),
        "dist": (aug - orig).abs().mean((1, 2, 3)),
        "hue_shift": dh.mean((1, 2)),
    }


def main():
    args = parse_args("diag_aug_parity", lambda ap: (
        ap.add_argument("--n", type=int, default=2000, help="images"),
        ap.add_argument("--repeats", type=int, default=5, help="augmentations per image")))
    seed_everything(args.seed)
    device = torch.device(args.device)

    split = data.get_splits("paper")["train"]
    idx = np.random.default_rng(args.seed).choice(len(split), args.n, replace=False)
    imgs = split.images[idx]                                         # (N, 32, 32, 3) uint8
    orig = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255)

    tv = transforms.Compose([transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(32, padding=C.CROP_PAD),
                             transforms.ColorJitter(**C.JITTER),
                             transforms.RandomRotation(C.ROTATION),
                             transforms.ToTensor()])
    gen = torch.Generator(device=device).manual_seed(args.seed)

    res = {"gpu": {}, "torchvision": {}}
    for _ in range(args.repeats):
        a_tv = torch.stack([tv(Image.fromarray(im)) for im in imgs])
        a_gpu = data.gpu_augment(orig.to(device), gen).cpu()
        for name, a in (("torchvision", a_tv), ("gpu", a_gpu)):
            for k, v in stats(a, orig).items():
                res[name].setdefault(k, []).append(v.numpy())

    report = {}
    print(f"{'stat':10s} {'torchvision':>18s} {'gpu':>18s}   KS     p")
    for k in res["gpu"]:
        a, b = np.concatenate(res["torchvision"][k]), np.concatenate(res["gpu"][k])
        ks = ks_2samp(a, b)
        report[k] = dict(tv_mean=float(a.mean()), tv_std=float(a.std()),
                         gpu_mean=float(b.mean()), gpu_std=float(b.std()),
                         ks=float(ks.statistic), p=float(ks.pvalue))
        flag = "  <-- differs" if ks.statistic > 0.05 else ""
        print(f"{k:10s} {a.mean():9.4f} ± {a.std():6.4f} {b.mean():9.4f} ± {b.std():6.4f}"
              f"   {ks.statistic:.3f}  {ks.pvalue:.1e}{flag}")

    out = run_dir(args) / "diag_aug_parity.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
