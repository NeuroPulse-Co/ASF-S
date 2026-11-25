"""
plot_results.py - Summarise results/prune_ratio.csv: accuracy vs sparsity per method.

One panel per fine-tuning setting (none / fc3-only / full), one line per method.
The five random seeds are collapsed into mean +/- one standard deviation, which tells
you how large a difference has to be before it means anything: two methods whose
curves sit inside the random band's width of each other are not distinguishable
with this data.

Run:  python plot_results.py            (after prune.py)
"""

import argparse
import os

import numpy as np
import pandas as pd

import config

STYLE = {"asf_fc3": ("tab:purple", "ASF-S (PG1 on fc3, consistent)"),
         "asf_fc2": ("tab:blue", "ASF-S (PG1 on fc2, consistent)"),
         "asf_as_coded": ("tab:gray", "ASF-S as coded (S=1, truncated)"),
         "l1": ("tab:orange", "L1 magnitude"),
         "l2": ("tab:green", "L2 magnitude"),
         "random": ("tab:red", "Random (mean ± sd, 5 seeds)")}


def load(path):
    df = pd.read_csv(path)
    # Collapse random_s0..random_s4 into one "random" method.
    df["family"] = df["method"].where(~df["method"].str.startswith("random"), "random")
    return df


def summary_table(df, cols):
    """Mean over seeds (only 'random' has several), per family and sparsity."""
    g = df.groupby(["family", "fc1_pruned_pct"])[cols]
    return g.mean().round(2), g.std().round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(config.RESULTS_DIR, "prune_ratio.csv"))
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "accuracy_vs_sparsity.png"))
    args = ap.parse_args()

    df = load(args.csv)
    settings = [c for c in ("acc_noft", "acc_head", "acc_full") if c in df]
    titles = {"acc_noft": "No fine-tuning", "acc_head": "Fine-tune fc3 only (paper's headline)",
              "acc_full": "Full fine-tuning"}
    mean, sd = summary_table(df, settings)
    base = df["base_acc"].iloc[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(settings), figsize=(5.5 * len(settings), 4.2), sharey=False)
    for ax, col in zip(np.atleast_1d(axes), settings):
        for fam, (color, label) in STYLE.items():
            if fam not in mean.index.get_level_values(0):
                continue
            x = mean.loc[fam].index.values
            y = mean.loc[fam][col].values
            ax.plot(x, y, "-o", ms=3, color=color, label=label, lw=1.5)
            if fam == "random":
                s = sd.loc[fam][col].values
                ax.fill_between(x, y - s, y + s, color=color, alpha=0.15)
        ax.axhline(base, ls="--", color="k", lw=0.8, label=f"unpruned ({base:.2f}%)")
        ax.set_title(titles[col]); ax.set_xlabel("fc1 neurons pruned (%)")
        ax.set_ylabel("eval accuracy (%)"); ax.grid(alpha=0.3)
        lo = mean[col].min()
        ax.set_ylim(max(lo - 1, 0), base + 0.5)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    print(f"Plot -> {args.out}\n")

    pd.set_option("display.width", 200)
    for col in settings:
        tab = mean[col].unstack(0)
        tab["random_sd"] = sd[col].unstack(0)["random"] if "random" in tab else np.nan
        print(f"== {titles[col]} (eval accuracy %, rows = % of fc1 pruned) ==")
        print(tab.to_string(), "\n")


if __name__ == "__main__":
    main()
