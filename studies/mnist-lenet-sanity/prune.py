"""
prune.py - STAGE 4: structurally prune fc1 neurons, fine-tune, and measure.

What "pruning fc1 neuron j" means physically
--------------------------------------------
    fc1: y1 = relu(W1 x + b1)      W1 is (300, 784): ROW j produces neuron j
    fc2: y2 = relu(W2 y1 + b2)     W2 is (100, 300): COLUMN j reads neuron j
Removing neuron j = delete row j of W1, entry j of b1 and column j of W2. Every
other weight is copied untouched, so the pruned network computes exactly what the
original would if neuron j always output 0. That makes "accuracy before fine-tuning"
a real measurement of the damage pruning did. (cifar100/pruning_utils.rebuild_fc
copied a column PREFIX instead, which rewires the layer; engineering-log #1.)

Experiment grid
---------------
  methods   importance files from importance.py (asf_*, l1, l2, random_s*)
  sparsity  --mode ratio     : keep the top-k neurons, k = round(300 * (1 - ratio)).
                               Every method is compared at EXACTLY the same sparsity.
            --mode threshold : keep neurons with score > thr (the paper's knob; the
                               resulting sparsity differs per method).
  fine-tune none : pruned model as is
            head : retrain fc3 only (fc1, fc2 frozen)  <- the paper's headline setting
            full : retrain all layers                   <- paper's ASF-S(a)

Data discipline: fine-tuning trains on MNIST train and early-stops on the ANALYSIS
split; every reported accuracy/ECE is on the untouched EVAL split.

Output: results/prune_<mode>.csv (rewritten each run, with a header).

Run:  python prune.py                              (full ratio grid)
      python prune.py --ratios 0.7 --ft head       (one quick setting)
"""

import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data import get_data, make_loader
from models import count_flops, count_params, load_lenet


# ── Structural pruning ────────────────────────────────────────────────────────

def prune_fc1(model, keep_idx):
    """Return a NEW model whose fc1 keeps only the neurons in keep_idx.
    Same logic as prune_linear() in git 6b81a13:CNN/cifar100_original_pipeline/
    pg_pruning_cifar100.py, which propagated indices fc1 -> fc2 correctly."""
    keep = torch.as_tensor(np.sort(keep_idx), dtype=torch.long)
    m = copy.deepcopy(model)
    old1, old2 = m.fc1, m.fc2
    m.fc1 = nn.Linear(old1.in_features, len(keep))
    m.fc2 = nn.Linear(len(keep), old2.out_features)
    with torch.no_grad():
        m.fc1.weight.copy_(old1.weight[keep, :])   # rows    -> which neurons exist
        m.fc1.bias.copy_(old1.bias[keep])
        m.fc2.weight.copy_(old2.weight[:, keep])   # columns -> what fc2 reads from them
        m.fc2.bias.copy_(old2.bias)                # fc2's own neurons are unchanged
    return m.eval()


def keep_by_ratio(scores, ratio):
    """Top-k neurons by score. Ties broken by index (stable sort) for reproducibility."""
    k = max(1, int(round(len(scores) * (1.0 - ratio))))
    return np.argsort(-scores, kind="stable")[:k]


def keep_by_threshold(scores, thr):
    keep = np.where(scores > thr)[0]
    return keep if len(keep) else np.array([int(np.argmax(scores))])  # never empty


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, x, y, n_bins=15):
    """Accuracy (%) and Expected Calibration Error.
    ECE: bin predictions by confidence; in each bin compare mean confidence with actual
    accuracy; average the gaps weighted by bin size. 0 = perfectly calibrated."""
    model.eval()
    prob = F.softmax(model(x), dim=1)
    conf, pred = prob.max(1)
    hit = (pred == y).float()
    ece = 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.float().mean() * (hit[in_bin].mean() - conf[in_bin].mean()).abs()
    return 100.0 * hit.mean().item(), float(ece)


def fine_tune(model, which, d, epochs, lr, patience, batch, seed):
    """Adam fine-tuning of the chosen layers with early stopping on the analysis split.
    The pre-fine-tuning weights count as 'epoch 0', so fine-tuning can never report a
    worse model than it started from. Returns (best model, epochs actually run)."""
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad = which == "full"
    if which == "head":
        for p in model.fc3.parameters():
            p.requires_grad = True
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loader = make_loader(d["train_x"], d["train_y"], batch, shuffle=True, seed=seed)

    best_acc, _ = evaluate(model, d["analysis_x"], d["analysis_y"])
    best_state, bad, ep = copy.deepcopy(model.state_dict()), 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        acc, _ = evaluate(model, d["analysis_x"], d["analysis_y"])
        if acc > best_acc:
            best_acc, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad = True
    return model, ep


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=config.CKPT_PATH)
    ap.add_argument("--mode", choices=["ratio", "threshold"], default="ratio")
    ap.add_argument("--ratios", type=float, nargs="+", default=config.RATIOS)
    ap.add_argument("--thresholds", type=float, nargs="+", default=config.THRESHOLDS)
    ap.add_argument("--methods", nargs="+",
                    default=["asf_as_coded", "asf_fc2", "asf_fc3", "l1", "l2"]
                            + [f"random_s{s}" for s in range(5)])
    ap.add_argument("--ft", nargs="+", choices=["head", "full"], default=["head", "full"])
    ap.add_argument("--epochs", type=int, default=config.FT_EPOCHS)
    ap.add_argument("--patience", type=int, default=config.FT_PATIENCE)
    ap.add_argument("--lr", type=float, default=config.FT_LR)
    ap.add_argument("--batch", type=int, default=config.FT_BATCH)
    ap.add_argument("--out", default=None, help="CSV path (default results/prune_<mode>.csv)")
    args = ap.parse_args()
    out = args.out or os.path.join(config.RESULTS_DIR, f"prune_{args.mode}.csv")

    d = get_data()
    base = load_lenet(args.ckpt)
    base_params, base_flops = count_params(base), count_flops(base)
    base_acc, base_ece = evaluate(base, d["eval_x"], d["eval_y"])
    print(f"Baseline: eval acc {base_acc:.2f}%  ECE {base_ece:.4f}  "
          f"params {base_params:,}  FLOPs {base_flops:,}")

    knobs = args.ratios if args.mode == "ratio" else args.thresholds
    rows = []
    for method in args.methods:
        if args.mode == "threshold" and method.startswith("random"):
            continue   # thresholds on uniform noise are just a ratio; skip
        scores = np.load(os.path.join(config.IMPORTANCE_DIR, f"{method}.npy"))
        for knob in knobs:
            keep = keep_by_ratio(scores, knob) if args.mode == "ratio" else keep_by_threshold(scores, knob)
            pruned = prune_fc1(base, keep)
            acc0, ece0 = evaluate(pruned, d["eval_x"], d["eval_y"])
            row = dict(method=method, mode=args.mode, knob=knob, kept_fc1=len(keep),
                       fc1_pruned_pct=100 * (1 - len(keep) / 300),
                       params=count_params(pruned),
                       param_red_pct=100 * (1 - count_params(pruned) / base_params),
                       flops_red_pct=100 * (1 - count_flops(pruned) / base_flops),
                       base_acc=base_acc, acc_noft=acc0, ece_noft=ece0)
            for which in args.ft:
                t0 = time.time()
                m, ep = fine_tune(copy.deepcopy(pruned), which, d, args.epochs, args.lr,
                                  args.patience, args.batch, config.SEED)
                acc, ece = evaluate(m, d["eval_x"], d["eval_y"])
                row.update({f"acc_{which}": acc, f"ece_{which}": ece, f"epochs_{which}": ep})
                row[f"sec_{which}"] = round(time.time() - t0, 1)
            rows.append(row)
            msg = "  ".join(f"{w}={row[f'acc_{w}']:.2f}" for w in args.ft)
            print(f"{method:13s} {args.mode}={knob:.2f} kept={len(keep):3d} "
                  f"params-{row['param_red_pct']:.1f}%  noFT={acc0:.2f}  {msg}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")  # crash-safe
    print(f"Results -> {out}")


if __name__ == "__main__":
    main()
