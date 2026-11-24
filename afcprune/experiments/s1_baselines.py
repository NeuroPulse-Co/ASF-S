"""
S1 – baseline factorial: selector × budget × recovery tier, for one dense checkpoint.

  python experiments/s1_baselines.py --dense PATH --tag NAME [--protocol clean|paper] [...]

For every selector (scores/ registry; `random` is drawn with --random-seeds) and every params↓ budget:
  1. allocation: the kept-count candidates within ±--tol of the budget (allocate.candidates), the
     --alloc-max nearest are pruned with the selector's top-k, BN-recalibrated, and scored on val accuracy;
     the best is kept. Same criterion for every selector (plans P5, P10). The Table 5 point (19, 16) is
     added as budget "t5pt" with a fixed allocation.
  2. recovery tiers on that pruned model (recovery.py):
       T0     none
       T1r    ridge head (target = dense logits, λ on val)          strict head-only, closed form
       T2r    BN recalibration + ridge head
       T3r    BN recal + LS compensation + BN recal + ridge head
       T1g    fc_frozen gradient FT                                 strict head-only, gradient
       T1fc   fc gradient FT with BN adapting                       = run01's mode (−10.50)
       T4     full FT
     Gradient tiers are slow; select with --tiers.
Everything is measured on val. Under --protocol paper val *is* the test set: the run is wrapped in
protocol.allow_test and logged (reproduction rows only, plan P4).

Outputs <outputs>/s1/<tag>/: results.csv (one row per selector × budget × tier, appended, finished rows
skipped on restart), alloc.json (every allocation table), contrast.npz + contrast.json (PG1/ΔG diagnostics),
gate.json (paper protocol: the run01 reproduction gate).
"""

import argparse
import contextlib
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from afcprune import allocate, config as C, data, metrics, models, paper_targets as P, pipeline, protocol  # noqa
from afcprune import recovery, surgery  # noqa: E402
from afcprune.scores import ScoreContext, get, names  # noqa: E402

CLOSED = ("T0", "T1r", "T2r", "T3r")
GRADIENT = ("T1g", "T1fc", "T4")
FIELDS = ["tag", "protocol", "selector", "budget", "tier", "ft_seed", "kept", "params", "flops", "params_red",
          "flops_red", "alloc_val_acc", "val_acc", "val_ece", "dense_val_acc", "d_acc", "ridge_lambda",
          "best_epoch", "epochs", "minutes"]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", required=True, help="state_dict .pth of the dense model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="conv2net", choices=list(models.MODELS))
    ap.add_argument("--protocol", default="clean", choices=["clean", "paper"])
    ap.add_argument("--outputs", default=str(C.OUTPUTS_DIR))
    ap.add_argument("--budgets", default=",".join(map(str, C.BUDGETS)) + ",t5pt")
    ap.add_argument("--selectors", default="pgi_aligned,pgi_literal,l1,l2,random,bn_gamma,fpgm,taylor,hrank,act_mean")
    ap.add_argument("--random-seeds", default="0,1,2")
    ap.add_argument("--tiers", default=",".join(CLOSED))
    ap.add_argument("--ft-seed", type=int, default=C.SEED)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--alloc-max", type=int, default=40)
    ap.add_argument("--bn-batches", type=int, default=20, help="BN recalibration batches (×500 images)")
    ap.add_argument("--calib-batches", type=int, default=20, help="taylor/act_mean/LS batches (×250 images)")
    ap.add_argument("--max-epochs", type=int, default=None, help="override FT epochs (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default=C.AUGMENT_BACKEND, choices=["gpu", "torchvision"])
    return ap.parse_args()


class Limited:
    """Re-iterable view of the first n batches of a loader."""
    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __iter__(self):
        for i, b in enumerate(self.loader):
            if i >= self.n:
                break
            yield b


def selector_list(args):
    out = []
    for s in args.selectors.split(","):
        if s == "random":
            out += [(f"random_s{r}", "random", int(r)) for r in args.random_seeds.split(",")]
        else:
            out.append((s, s, args.ft_seed))
    unknown = {base for _, base, _ in out} - set(names())
    if unknown:
        raise SystemExit(f"unknown selectors {sorted(unknown)}; registered: {names()}")
    return out


def done_keys(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {(r["selector"], r["budget"], r["tier"], int(r["ft_seed"])) for r in csv.DictReader(f)}


def append(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def val_metrics(model, val):
    logits, labels = metrics.predict(model, val)
    return metrics.accuracy(logits, labels), metrics.ece(logits, labels)


def main():
    args = parse()
    torch.manual_seed(args.ft_seed); np.random.seed(args.ft_seed)
    device = torch.device(args.device)
    out = Path(args.outputs) / "s1" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    csv_path, alloc_path = out / "results.csv", out / "alloc.json"
    done = done_keys(csv_path)
    alloc_cache = json.loads(alloc_path.read_text()) if alloc_path.exists() else {}
    guard = (protocol.allow_test(f"S1 paper-protocol reproduction row ({args.tag}): val = test", out)
             if args.protocol == "paper" else contextlib.nullcontext())
    t_start = time.time()

    with guard:
        splits = protocol.get_splits(args.protocol)
        train, val_split, pg = splits.train, splits.val, splits.pg
        L = lambda s, bs, aug, seed=args.ft_seed: data.loader(s, bs, aug, device, args.backend, seed)  # noqa
        val = L(val_split, 1000, False)
        train_plain = L(train, 500, False)                      # ridge / BN recal / LS (no augmentation)
        calib = Limited(L(train, 250, False), args.calib_batches)

        dense = models.build(args.model).to(device)
        dense.load_state_dict(torch.load(args.dense, map_location=device, weights_only=True))
        dense.eval()
        dense_acc, dense_ece = val_metrics(dense, val)
        cpu = copy.deepcopy(dense).cpu()
        base_params, base_flops = metrics.count_params(cpu), metrics.count_flops(cpu)
        print(f"[{args.tag}] dense val acc {dense_acc:.2f}  ECE {dense_ece:.4f}  params {base_params}  "
              f"FLOPs {base_flops}", flush=True)

        # output-layer AFC → PCA components (for PGI) + P3 diagnostics
        con = pipeline.output_contrast(dense, L(pg, 500, False), n_boot=30)
        np.savez_compressed(out / "contrast.npz", **{k: v for k, v in con.items() if k != "idx"})
        diag = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in con.items()
                if k in ("sigma", "gap12_min", "gap23_min", "gap23_median", "topk_explained", "pg_accuracy",
                         "counts", "boot_median_abs_cos")}
        (out / "contrast.json").write_text(json.dumps(diag, indent=1))
        print(f"  contrast: σ {con['sigma']:.3f}  top-{C.PCA_K} var {con['topk_explained']:.3f}  "
              f"min λ2−λ3 {con['gap23_min']:.4f}  min group {con['counts'].min()}  "
              f"boot |cos| median-of-medians {np.median(con['boot_median_abs_cos']):.3f}", flush=True)

        budgets = [b for b in args.budgets.split(",")]
        tiers = args.tiers.split(",")
        for label, base, seed in selector_list(args):
            ctx = ScoreContext(device=device, calib=calib, seed=seed,
                               extra={"components": con["comps"], "hrank_batches": 5})
            t0 = time.time()
            scores = get(base)(dense, ctx)
            print(f"\n[{label}] scored in {time.time() - t0:.1f}s", flush=True)
            for b in budgets:
                todo = [t for t in tiers if (label, b, t, args.ft_seed) not in done]
                if not todo:
                    continue
                key = f"{label}|{b}"
                if key not in alloc_cache:
                    if b == "t5pt":
                        kept = dict(P.TABLE5_ASFS_KEPT[args.model])
                        table = []
                        pm = surgery.prune(dense, allocate.keep_from_counts(scores, kept))
                        recovery.bn_recalibrate(pm, train_plain, args.bn_batches)
                        a_acc = val_metrics(pm, val)[0]
                    else:
                        cands = allocate.candidates(dense, float(b), args.tol)
                        if not cands:
                            print(f"  budget {b}: no candidate within ±{args.tol}", flush=True)
                            continue

                        def crit(keep):
                            pm = surgery.prune(dense, keep)
                            recovery.bn_recalibrate(pm, train_plain, args.bn_batches)
                            return val_metrics(pm, val)[0]
                        kept, _, table = allocate.choose(scores, cands, crit, args.alloc_max)
                        a_acc = max(t[2] for t in table)
                    alloc_cache[key] = dict(kept=kept, alloc_val_acc=a_acc,
                                            table=[[k, r, s] for k, r, s in table])
                    alloc_path.write_text(json.dumps(alloc_cache))
                kept, a_acc = alloc_cache[key]["kept"], alloc_cache[key]["alloc_val_acc"]
                keep = allocate.keep_from_counts(scores, kept)
                pcpu = surgery.prune(cpu, keep)
                params, flops = metrics.count_params(pcpu), metrics.count_flops(pcpu)
                for tier in todo:
                    t1 = time.time()
                    pm = surgery.prune(dense, keep)
                    lam = best_epoch = epochs = ""
                    if tier in ("T2r", "T3r"):
                        recovery.bn_recalibrate(pm, train_plain, args.bn_batches)
                    if tier == "T3r":
                        recovery.ls_compensate(dense, pm, keep, calib, args.calib_batches)
                        recovery.bn_recalibrate(pm, train_plain, args.bn_batches)
                    if tier in ("T1r", "T2r", "T3r"):
                        lam = recovery.ridge_head(pm, train_plain, val, C.RIDGE_LAMBDAS, target="logits",
                                                  teacher=dense)["lambda"]
                    if tier in GRADIENT:
                        mode = {"T1g": "fc_frozen", "T1fc": "fc", "T4": "full"}[tier]
                        torch.manual_seed(args.ft_seed)
                        cfg = C.FINETUNE
                        best, hist = recovery.finetune(pm, mode, L(train, cfg["batch_size"], True), val,
                                                       cfg["lr"], args.max_epochs or cfg["max_epochs"],
                                                       cfg["patience"])
                        best_epoch, epochs = best["epoch"], len(hist)
                    acc, e = val_metrics(pm, val)
                    row = dict(tag=args.tag, protocol=args.protocol, selector=label, budget=b, tier=tier,
                               ft_seed=args.ft_seed, kept="/".join(str(kept[c]) for c in kept), params=params,
                               flops=flops, params_red=round(100 * (1 - params / base_params), 2),
                               flops_red=round(100 * (1 - flops / base_flops), 2),
                               alloc_val_acc=round(a_acc, 2), val_acc=round(acc, 2), val_ece=round(e, 4),
                               dense_val_acc=round(dense_acc, 2), d_acc=round(acc - dense_acc, 2),
                               ridge_lambda=lam, best_epoch=best_epoch, epochs=epochs,
                               minutes=round((time.time() - t1) / 60, 2))
                    append(csv_path, row)
                    print(f"  {b:>5} {tier:4s} kept {row['kept']:>7}  params↓ {row['params_red']:5.1f}  "
                          f"val {acc:5.2f}  Δ {row['d_acc']:+6.2f}  ({row['minutes']} min)", flush=True)

        if args.protocol == "paper" and csv_path.exists():
            with open(csv_path) as f:
                rows = [r for r in csv.DictReader(f) if r["selector"] == "pgi_aligned" and r["budget"] == "t5pt"
                        and r["tier"] == "T1fc"]
            if rows:
                d = float(rows[-1]["d_acc"])
                gate = dict(check="run01 reproduction: pgi_aligned @ (19,16), T1fc ΔAcc = −10.50 ± 0.5",
                            ours=d, ref=-10.50, tol=0.5, passed=abs(d + 10.50) <= 0.5)
                (out / "gate.json").write_text(json.dumps(gate, indent=1))
                print(f"\nGATE {'PASS' if gate['passed'] else 'FAIL'}: {gate}")
    print(f"\n[{args.tag}] done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
