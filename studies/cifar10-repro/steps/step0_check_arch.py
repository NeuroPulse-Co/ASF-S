"""
Step 0 – architecture, data and throughput check (no training).

Gate [paper Table 4]:
  params == 183K (Conv2Net) / 815K (Conv6Net), within 0.5%
  FLOPs within 1% of 40.89M / 79.47M
  split sizes match the protocol
Info: seconds per training epoch for the gpu and torchvision backends.
"""

import time

import torch
import torch.nn as nn

from common import C, check, parse_args, seed_everything, step_dir, write_gate
import data
import models
import paper_targets as P
from asfs.metrics import count_flops, count_params


def epoch_seconds(model, split, backend, device, max_batches=None):
    """Time one training epoch (forward + backward + Adam step), extrapolated if truncated."""
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    dl = data.loader(split, C.TRAIN["conv2net"]["batch_size"], True, device, backend)
    n_batches = len(dl) if max_batches is None else min(max_batches, len(dl))
    it = iter(dl)
    next(it)                                                   # warm-up (workers, cuDNN)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_batches - 1):
        x, y = next(it)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) * len(dl) / (n_batches - 1)


def main():
    args = parse_args("step0_check_arch", lambda ap: ap.add_argument(
        "--bench-batches", type=int, default=60,
        help="batches timed for the torchvision backend (extrapolated to an epoch)"))
    seed_everything(args.seed)
    device = torch.device(args.device)
    out = step_dir(args, "step0_check_arch")

    model = models.build(args.model)
    params, flops = count_params(model), count_flops(model)
    ref = P.DENSE[args.model]

    splits = data.get_splits(args.protocol)
    sizes = {k: len(v) for k, v in splits.items()}
    want = ({"train": 50000, "val": 10000, "test": 10000, "pg": 10 * C.ACT_PER_CLASS}
            if args.protocol == "paper" else
            {"train": 50000 - C.VAL_SIZE, "val": C.VAL_SIZE, "test": 10000, "pg": C.VAL_SIZE})

    checks = [
        check("params", params, ref["params"], tolerance=round(0.005 * ref["params"])),
        check("flops", flops, ref["flops"], tolerance=round(0.01 * ref["flops"])),
        check("split_sizes", sizes, want, ok=sizes == want),
    ]

    bench = {}
    if device.type == "cuda":
        bench["gpu_s_per_epoch"] = round(epoch_seconds(models.build(args.model), splits["train"],
                                                       "gpu", device), 1)
    bench["torchvision_s_per_epoch"] = round(epoch_seconds(
        models.build(args.model), splits["train"], "torchvision", device, args.bench_batches), 1)
    print("throughput:", bench)

    write_gate(out, "step0_check_arch", args, checks,
               info={"params": params, "flops": flops, "throughput": bench,
                     "torch": torch.__version__, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None})


if __name__ == "__main__":
    main()
