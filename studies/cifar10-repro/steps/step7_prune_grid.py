"""
Step 7 – one-shot pruning over the per-layer τ grid, before any fine-tuning [paper §4, Eq. 14, §B.2].

For every (τ1, τ2) in THRESHOLDS × THRESHOLDS (Conv2Net: conv1, conv2), filters with
PGI < τ are removed physically. Per grid point: kept filters, params and FLOPs (and their
reduction), and accuracy *before* fine-tuning on the val split (paper protocol: test).

At the same kept counts, two cheap baselines are scored too: L1-norm (largest kept) and
random (mean of 5 draws). This is an early, fine-tuning-free answer to "does PGI pick
better filters?" (on cifar_100 it did not).

Gate:
  1. prune == mask: the pruned network matches its full-size masked twin (max |Δlogit| < 1e-3)
     on 3 grid points
  2. the paper's operating points exist in the grid: a point within ±5 of 70% parameter
     reduction (Table 1) and within ±5 of 75.8% (Table 5)
Info: FLOPs reduction at those points vs the paper's 57.01% and 82.7%, and the sparsity
definition question (parameters vs filters).
Outputs: grid.csv, fig_grid.png, gate.json.
"""

import copy
import csv
import itertools

import numpy as np
import torch

from common import C, check, parse_args, run_dir, seed_everything, step_dir, write_gate
import data
import models
import paper_targets as P
from asfs.metrics import accuracy, count_flops, count_params, predict
from asfs.prune import keep_from_scores, keep_top_k, kept_counts, masked, prune

RANDOM_DRAWS = 5


def load_pgi(args, variant):
    z = np.load(run_dir(args) / "step6_pgi" / "pgi.npz")
    return {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"{variant}__")}


def evaluate(model, dl):
    logits, labels = predict(model, dl)
    return accuracy(logits, labels)


def stats(model):
    cpu = copy.deepcopy(model).cpu()
    return count_params(cpu), count_flops(cpu)


def nearest(rows, key, target):
    return min(rows, key=lambda r: abs(r[key] - target))


def main():
    args = parse_args("step7_prune_grid", lambda ap: ap.add_argument(
        "--variant", default="primary", choices=["primary", "aligned", "original"],
        help="which step-6 PGI to prune by"))
    seed_everything(args.seed)
    device = torch.device(args.device)
    out = step_dir(args, f"step7_prune_grid_{args.variant}")

    model = models.build(args.model).to(device)
    model.load_state_dict(torch.load(run_dir(args) / "step1_train" / "model_best.pth",
                                     map_location=device, weights_only=True))
    model.eval()
    val = data.loader(data.get_splits(args.protocol)["val"], 1000, False, device, args.backend)
    base_acc = evaluate(model, val)
    base_params, base_flops = stats(model)

    pgi = load_pgi(args, args.variant)
    layers = [c for c, _, _ in model.CONV_SEQ]
    l1 = {c: getattr(model, c).weight.detach().abs().sum((1, 2, 3)).cpu().numpy() for c in layers}
    rng = np.random.default_rng(args.seed)
    print(f"dense: acc {base_acc:.2f}  params {base_params}  FLOPs {base_flops}")

    if len(layers) == 2:
        grid = list(itertools.product(C.THRESHOLDS, repeat=2))
    else:                                     # deeper models: one τ shared by all layers
        grid = [(t,) * len(layers) for t in C.THRESHOLDS]

    rows, equiv = [], []
    for taus in grid:
        keep = {c: keep_from_scores(pgi[c], t) for c, t in zip(layers, taus)}
        pm = prune(model, keep)
        params, flops = stats(pm)
        acc = evaluate(pm, val)
        if len(equiv) < 3 and taus[0] in (0.2, 0.5, 0.8) and taus[0] == taus[-1]:
            x = next(iter(val))[0][:256]
            with torch.no_grad():
                equiv.append(float((pm(x) - masked(model, keep)(x)).abs().max()))
        counts = kept_counts(keep)
        acc_l1 = evaluate(prune(model, {c: keep_top_k(l1[c], counts[c]) for c in layers}), val)
        acc_rand = np.mean([evaluate(prune(model, {c: keep_top_k(rng.random(len(l1[c])), counts[c])
                                                   for c in layers}), val) for _ in range(RANDOM_DRAWS)])
        filters_total = sum(len(pgi[c]) for c in layers)
        row = {**{f"tau_{c}": t for c, t in zip(layers, taus)},
               **{f"kept_{c}": counts[c] for c in layers},
               "params": params, "flops": flops,
               "params_red": round(100 * (1 - params / base_params), 2),
               "flops_red": round(100 * (1 - flops / base_flops), 2),
               "filters_red": round(100 * (1 - sum(counts.values()) / filters_total), 2),
               "pre_acc": round(acc, 2), "pre_acc_l1": round(acc_l1, 2), "pre_acc_random": round(float(acc_rand), 2)}
        rows.append(row)
        print("  " + " ".join(f"{c}={t}" for c, t in zip(layers, taus)) +
              f"  kept {list(counts.values())}  params↓ {row['params_red']:5.1f}%  FLOPs↓ {row['flops_red']:5.1f}%"
              f"  pre-acc PGI {row['pre_acc']:5.2f}  L1 {row['pre_acc_l1']:5.2f}  rand {row['pre_acc_random']:5.2f}")

    with open(out / "grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # unique kept-count configurations only, so duplicate grid points do not double count
    uniq = {tuple(r[f"kept_{c}"] for c in layers): r for r in rows}.values()
    beats_l1 = np.mean([r["pre_acc"] > r["pre_acc_l1"] for r in uniq])
    beats_rand = np.mean([r["pre_acc"] > r["pre_acc_random"] for r in uniq])

    t1, t5 = P.TABLE1_ASFS[args.model], P.TABLE5[args.model]["asfs"]
    p70, p76 = nearest(rows, "params_red", t1["sparsity"]), nearest(rows, "params_red", t5["params_red"])
    print(f"\nnearest to Table 1 (70% sparsity): params↓ {p70['params_red']}%  FLOPs↓ {p70['flops_red']}% "
          f"(paper {t1['flops_reduction']}%)  filters↓ {p70['filters_red']}%")
    print(f"nearest to Table 5 (75.8% params↓): params↓ {p76['params_red']}%  FLOPs↓ {p76['flops_red']}% "
          f"(paper {t5['flops_red']}%)")
    print(f"pre-FT, unique kept configs: PGI beats L1 in {beats_l1:.0%}, beats random in {beats_rand:.0%}")
    plot(rows, layers, out / "fig_grid.png")

    checks = [
        check("prune == mask (max |Δlogit|)", round(max(equiv), 6), 0.0, tolerance=1e-3,
              ok=len(equiv) > 0 and max(equiv) < 1e-3),
        check("a grid point within ±5 of 70% params↓ (Table 1)", p70["params_red"], t1["sparsity"], tolerance=5),
        check("a grid point within ±5 of 75.8% params↓ (Table 5)", p76["params_red"], t5["params_red"], tolerance=5),
    ]
    write_gate(out, "step7_prune_grid", args, checks, info={
        "variant": args.variant, "base_acc": round(base_acc, 2), "base_params": base_params, "base_flops": base_flops,
        "near_table1": p70, "near_table5": p76, "unique_configs": len(uniq),
        "pre_ft_pgi_beats_l1": round(float(beats_l1), 3), "pre_ft_pgi_beats_random": round(float(beats_rand), 3)})


def plot(rows, layers, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["params_red"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, [r["pre_acc"] for r in rows], label="PGI", s=18)
    ax.scatter(x, [r["pre_acc_l1"] for r in rows], label="L1 (same kept counts)", s=10, marker="x")
    ax.scatter(x, [r["pre_acc_random"] for r in rows], label="random (mean of 5)", s=10, marker="+")
    ax.set_xlabel("parameter reduction (%)"); ax.set_ylabel("accuracy before fine-tuning (%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("τ grid, one-shot pruning, no fine-tuning", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
