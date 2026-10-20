"""
Step 8 – fine-tune pruned models [paper §5.3–5.4, Tables 1 and 5, Fig. 6].

--scope paper  The paper's Table 5 operating point: exactly (19, 16) kept filters
               (paper_targets.TABLE5_ASFS_KEPT, derived in step 7). Filters chosen as the
               top-k by: aligned PGI, original PGI (the authors' formula), L1, L2, and random
               (3 draws). Each pruned model is fine-tuned in modes fc, fc_frozen and full
               (asfs/finetune.py). 21 runs, ~1–1.5 h.
--scope grid   Every unique kept-count configuration of step 7's τ grid (--grid-variant,
               default aligned), fine-tuned in mode fc. Fig. 6-style Pareto plot. ~2.5 h.
--scope all    Both (default).

Gate [paper Table 5: ASF-S ΔAcc −3.78 at 75.8% params↓], on the paper scope, mode fc,
aligned PGI (config.PGI_MODE), with the original PGI reported alongside:
  1. ΔAcc ≥ paper − 2 (one-sided: a smaller drop than the paper's is not a failure)
  2. PGI ≥ L1 and 3. PGI ≥ mean of random (the paper's claim that PGI is the better criterion)

--ft-seeds 42,43,44 repeats each fine-tune with those seeds (pruning selection unchanged), to
separate method differences from run-to-run noise; --methods restricts which selections run.
Tables and the gate use the mean over seeds.

Results are appended to results.csv after each run, and finished runs (scope, method, mode,
kept counts, ft_seed) are skipped on restart, so a killed session loses at most one run.
Outputs: results.csv, fig6_pareto.png, gate.json.
"""

import copy
import csv

import numpy as np
import torch

from common import C, check, parse_args, run_dir, seed_everything, step_dir, write_gate
import data
import models
import paper_targets as P
from asfs.finetune import finetune
from asfs.metrics import accuracy, count_flops, count_params, ece, predict
from asfs.prune import keep_top_k, prune

FIELDS = ["scope", "method", "ft_mode", "ft_seed", "kept_conv1", "kept_conv2", "params", "flops", "params_red",
          "flops_red", "pre_acc", "best_val_acc", "best_epoch", "epochs", "test_acc", "test_ece",
          "d_acc", "minutes"]


def scores_by_method(model, args):
    z = np.load(run_dir(args) / "step6_pgi" / "pgi.npz")
    layers = [c for c, _, _ in model.CONV_SEQ]
    s = {"pgi_aligned": {c: z[f"aligned__{c}"] for c in layers},
         "pgi_original": {c: z[f"original__{c}"] for c in layers}}
    w = {c: getattr(model, c).weight.detach().cpu() for c in layers}
    s["l1"] = {c: w[c].abs().sum((1, 2, 3)).numpy() for c in layers}
    s["l2"] = {c: w[c].pow(2).sum((1, 2, 3)).sqrt().numpy() for c in layers}
    rng = np.random.default_rng(args.seed)
    for d in range(args.random_draws):
        s[f"random{d}"] = {c: rng.random(len(s["l1"][c])) for c in layers}
    return s


def migrate(path):
    """results.csv from before --ft-seeds has no ft_seed column; those runs used seed C.SEED."""
    if not path.exists():
        return
    with open(path) as f:
        reader = csv.DictReader(f)
        if "ft_seed" in reader.fieldnames:
            return
        rows = [{**r, "ft_seed": C.SEED} for r in reader]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"migrated {len(rows)} rows of {path.name}: added ft_seed = {C.SEED}")


def key(scope, method, mode, ft_seed, k1, k2):
    return (scope, method, mode, int(ft_seed), int(k1), int(k2))


def done_keys(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {key(r["scope"], r["method"], r["ft_mode"], r["ft_seed"], r["kept_conv1"], r["kept_conv2"])
                for r in csv.DictReader(f)}


def append(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(args, model, keep, ft_mode, ft_seed, splits, device, base_acc, base_params, base_flops):
    pm = prune(model, keep)
    cpu = copy.deepcopy(pm).cpu()
    params, flops = count_params(cpu), count_flops(cpu)
    val = data.loader(splits["val"], 1000, False, device, args.backend)
    test = data.loader(splits["test"], 1000, False, device, args.backend)
    pre_acc = accuracy(*predict(pm, val))

    seed_everything(ft_seed)
    cfg = C.FINETUNE
    train = data.loader(splits["train"], cfg["batch_size"], True, device, args.backend, ft_seed)
    best, hist = finetune(pm, ft_mode, train, val, cfg["lr"], args.max_epochs or cfg["max_epochs"],
                          cfg["patience"])
    logits, labels = predict(pm, test)
    test_acc = accuracy(logits, labels)
    return {"kept_conv1": int(keep["conv1"].sum()), "kept_conv2": int(keep["conv2"].sum()),
            "params": params, "flops": flops,
            "params_red": round(100 * (1 - params / base_params), 2),
            "flops_red": round(100 * (1 - flops / base_flops), 2),
            "pre_acc": round(pre_acc, 2), "best_val_acc": round(best["val_acc"], 2),
            "best_epoch": best["epoch"], "epochs": len(hist), "test_acc": round(test_acc, 2),
            "test_ece": round(ece(logits, labels), 4), "d_acc": round(test_acc - base_acc, 2),
            "minutes": round(sum(h["seconds"] for h in hist) / 60, 1)}


def plot(rows, base_acc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    g = [r for r in rows if r["scope"] == "grid"]
    if g:
        ax.scatter([float(r["params_red"]) for r in g], [float(r["test_acc"]) for r in g],
                   s=16, c="tab:blue", label="ASF-S τ grid (fc)")
    marks = {"pgi_aligned": ("*", "tab:red"), "pgi_original": ("*", "tab:orange"),
             "l1": ("s", "tab:green"), "l2": ("D", "tab:olive"), "random": ("x", "gray")}
    for r in rows:
        if r["scope"] != "paper" or r["ft_mode"] != "fc":
            continue
        key = "random" if r["method"].startswith("random") else r["method"]
        m, col = marks[key]
        ax.scatter(float(r["params_red"]), float(r["test_acc"]), marker=m, c=col, s=70,
                   label=f"{key} @ (19,16), fc")
    ax.axhline(base_acc, ls="--", c="k", lw=0.8, label=f"dense {base_acc:.2f}")
    ax.axhline(base_acc + P.TABLE5["conv2net"]["asfs"]["d_acc"], ls=":", c="tab:red", lw=0.8,
               label="dense − 3.78 (paper Table 5)")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7)
    ax.set_xlabel("parameter reduction (%)"); ax.set_ylabel("test accuracy after fine-tuning (%)")
    ax.grid(alpha=0.3); ax.set_title("Accuracy vs sparsity after fine-tuning – cf. paper Fig. 6", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    def extra(ap):
        ap.add_argument("--scope", default="all", choices=["paper", "grid", "all"])
        ap.add_argument("--grid-variant", default="aligned", choices=["aligned", "original", "primary"])
        ap.add_argument("--ft-modes", default="fc,fc_frozen,full", help="paper scope fine-tuning modes")
        ap.add_argument("--random-draws", type=int, default=3)
        ap.add_argument("--ft-seeds", default=None, help="comma-separated fine-tuning seeds (default: --seed)")
        ap.add_argument("--methods", default=None,
                        help="comma-separated subset, e.g. pgi_aligned,pgi_original,l1,l2 (default: all)")
        ap.add_argument("--max-epochs", type=int, default=None, help="override (smoke tests)")
    args = parse_args("step8_finetune", extra)
    if args.model != "conv2net":
        raise SystemExit("step 8 is written for Conv2Net (two conv layers, Table 5 kept counts)")
    seed_everything(args.seed)
    device = torch.device(args.device)
    out = step_dir(args, "step8_finetune")
    csv_path = out / "results.csv"
    migrate(csv_path)
    done = done_keys(csv_path)
    ft_seeds = [int(x) for x in (args.ft_seeds or str(args.seed)).split(",")]

    model = models.build(args.model).to(device)
    model.load_state_dict(torch.load(run_dir(args) / "step1_train" / "model_best.pth",
                                     map_location=device, weights_only=True))
    model.eval()
    splits = data.get_splits(args.protocol)
    base_acc = accuracy(*predict(model, data.loader(splits["test"], 1000, False, device, args.backend)))
    cpu = copy.deepcopy(model).cpu()
    base_params, base_flops = count_params(cpu), count_flops(cpu)
    layers = [c for c, _, _ in model.CONV_SEQ]
    scores = scores_by_method(model, args)
    methods = list(scores) if args.methods is None else args.methods.split(",")
    unknown = set(methods) - set(scores)
    if unknown:
        raise SystemExit(f"unknown --methods {sorted(unknown)}; choose from {list(scores)}")
    print(f"dense test acc {base_acc:.2f}   results → {csv_path}")

    jobs = []
    if args.scope in ("paper", "all"):
        k = P.TABLE5_ASFS_KEPT[args.model]
        for method in methods:
            for mode in args.ft_modes.split(","):
                for sd in ft_seeds:
                    jobs.append(("paper", method, mode, sd,
                                 {c: keep_top_k(scores[method][c], k[c]) for c in layers}))
    if args.scope in ("grid", "all"):
        with open(run_dir(args) / f"step7_prune_grid_{args.grid_variant}" / "grid.csv") as f:
            counts = sorted({(int(r["kept_conv1"]), int(r["kept_conv2"])) for r in csv.DictReader(f)})
        pgi = scores[f"pgi_{args.grid_variant}"] if args.grid_variant != "primary" else scores["pgi_aligned"]
        for k1, k2 in counts:
            for sd in ft_seeds:
                jobs.append(("grid", f"pgi_{args.grid_variant}", "fc", sd,
                             {"conv1": keep_top_k(pgi["conv1"], k1), "conv2": keep_top_k(pgi["conv2"], k2)}))

    todo = [j for j in jobs if key(*j[:4], j[4]["conv1"].sum(), j[4]["conv2"].sum()) not in done]
    print(f"{len(jobs)} runs planned, {len(jobs) - len(todo)} already done, {len(todo)} to go")
    for i, (scope, method, mode, sd, keep) in enumerate(todo, 1):
        r = run_one(args, model, keep, mode, sd, splits, device, base_acc, base_params, base_flops)
        row = {"scope": scope, "method": method, "ft_mode": mode, "ft_seed": sd, **r}
        append(csv_path, row)
        print(f"[{i}/{len(todo)}] {scope:5s} {method:13s} {mode:9s} seed {sd} kept ({r['kept_conv1']},{r['kept_conv2']}) "
              f"params↓ {r['params_red']:5.1f}%  pre {r['pre_acc']:5.2f} → test {r['test_acc']:5.2f} "
              f"(Δ {r['d_acc']:+.2f}, ep {r['best_epoch']}/{r['epochs']}, {r['minutes']} min)", flush=True)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    plot(rows, base_acc, out / "fig6_pareto.png")

    runs = {}
    for r in rows:
        if r["scope"] == "paper":
            runs.setdefault((r["method"], r["ft_mode"]), []).append(float(r["d_acc"]))
    if not runs:
        return
    paper = {k: float(np.mean(v)) for k, v in runs.items()}
    print("\nPaper point (19, 16) — ΔAcc vs dense, mean ± sd over fine-tuning seeds (n):")
    all_methods = sorted({m for m, _ in paper})
    modes = sorted({md for _, md in paper})
    print(f"  {'method':14s}" + "".join(f"{md:>20s}" for md in modes))
    for m in all_methods:
        cells = []
        for md in modes:
            v = runs.get((m, md))
            cells.append(f"{np.mean(v):7.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:4.2f} ({len(v)})"
                         if v else f"{'':>20s}")
        print(f"  {m:14s}" + "".join(f"{c:>20s}" for c in cells))

    target = P.TABLE5[args.model]["asfs"]["d_acc"]
    primary = f"pgi_{C.PGI_MODE}" if C.PGI_MODE in ("aligned", "original") else "pgi_aligned"
    rand = np.mean([v for (m, md), v in paper.items() if m.startswith("random") and md == "fc"])
    ours = paper.get((primary, "fc"))
    checks = []
    if ours is not None:
        checks = [
            check(f"{primary} fc ΔAcc ≥ paper − 2", ours, target, tolerance=2, ok=ours >= target - 2),
            check(f"{primary} fc ΔAcc ≥ L1", ours, paper.get(("l1", "fc")), ok=ours >= paper.get(("l1", "fc"), np.inf)),
            check(f"{primary} fc ΔAcc ≥ random mean", ours, round(float(rand), 2), ok=ours >= rand),
        ]
    write_gate(out, "step8_finetune", args, checks, info={
        "base_test_acc": round(base_acc, 2), "paper_target_d_acc": target,
        "paper_point_mean": {f"{m}|{md}": round(v, 2) for (m, md), v in paper.items()},
        "paper_point_runs": {f"{m}|{md}": v for (m, md), v in runs.items()},
        "random_fc_mean": round(float(rand), 2),
        "grid_runs": sum(r["scope"] == "grid" for r in rows)})


if __name__ == "__main__":
    main()
