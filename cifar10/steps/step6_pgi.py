"""
Step 6 – PGI score per conv filter [paper §4 Eq. 12–13, App. B.1 Fig. 11].

Uses the step-1 model and the step-5 PCA components. Computes three variants:
  primary   config.PGI_MODE / ABS_OF_MEAN (default: aligned, Eq. 13's mean_k |s_ik|)
  aligned   always, Eq. 13 as written
  original  literal + |mean_k s_ik|: the authors' min_scripts computation

The paper gives no number for this step; Fig. 11 only shows per-layer distributions. The
gate is therefore a sanity check, and the informative output is diagnostic:
  gate: every layer's PGI is finite, spans [0, 1] (min–max), and is not degenerate
        (≥ half the filters have distinct scores)
  info: per layer, the fraction of filters below each τ ∈ {0.1, …, 0.9} (what step 7's
        threshold grid will prune); Spearman ρ of PGI vs the filter's L1 norm (does PGI
        say more than weight magnitude?) and between the aligned and original variants
Outputs: pgi.npz, fig11_pgi.png, gate.json.
"""

import numpy as np
import torch
from scipy.stats import spearmanr

from common import C, check, parse_args, run_dir, seed_everything, step_dir, write_gate
import models
from asfs.pgi import pgi, pgi_aligned, pgi_literal


def l1_norms(model):
    return {conv: getattr(model, conv).weight.detach().abs().sum((1, 2, 3)).cpu().numpy()
            for conv, _, _ in model.CONV_SEQ}


def plot(variants, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = list(next(iter(variants.values())))
    fig, ax = plt.subplots(len(variants), len(layers), figsize=(5 * len(layers), 2.6 * len(variants)),
                           squeeze=False)
    for r, (name, v) in enumerate(variants.items()):
        for c, layer in enumerate(layers):
            a = ax[r, c]
            a.bar(np.arange(len(v[layer])), np.sort(v[layer]), color="tab:green")
            for t in C.THRESHOLDS:
                a.axhline(t, color="gray", lw=0.3)
            a.set_title(f"{name}: {layer} ({len(v[layer])} filters)", fontsize=9)
            a.set_ylim(0, 1.02); a.set_xticks([])
    fig.suptitle("Normalised PGI per filter, sorted – cf. paper Fig. 11", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    args = parse_args("step6_pgi")
    seed_everything(args.seed)
    out = step_dir(args, "step6_pgi")

    model = models.build(args.model)
    model.load_state_dict(torch.load(run_dir(args) / "step1_train" / "model_best.pth",
                                     map_location="cpu", weights_only=True))
    model.eval()
    comps = np.load(run_dir(args) / "step5_contrast_pca" / "contrast.npz")["components"]

    variants = {
        "primary": pgi(model, comps, C.PGI_MODE, C.ABS_OF_MEAN),
        "aligned": pgi_aligned(model, comps, abs_of_mean=False),
        "original": pgi_literal(model, comps, abs_of_mean=True),
    }
    np.savez_compressed(out / "pgi.npz", **{f"{v}__{layer}": s for v, d in variants.items() for layer, s in d.items()},
                        mode=C.PGI_MODE, abs_of_mean=C.ABS_OF_MEAN)

    l1 = l1_norms(model)
    info, checks = {"mode": C.PGI_MODE, "abs_of_mean": C.ABS_OF_MEAN}, []
    for layer, p in variants["primary"].items():
        below = {t: round(float((p < t).mean()), 3) for t in C.THRESHOLDS}
        rho_l1 = spearmanr(p, l1[layer])[0]
        rho_orig = spearmanr(variants["aligned"][layer], variants["original"][layer])[0]
        n_distinct = len(np.unique(np.round(p, 6)))
        print(f"\n{layer}: {len(p)} filters, {n_distinct} distinct scores")
        print("  fraction below τ: " + "  ".join(f"{t}:{v:.2f}" for t, v in below.items()))
        print(f"  Spearman ρ  PGI vs L1 norm {rho_l1:+.3f}   aligned vs original {rho_orig:+.3f}")
        info[layer] = {"filters": len(p), "fraction_below": below,
                       "spearman_vs_l1": round(float(rho_l1), 3),
                       "spearman_aligned_vs_original": round(float(rho_orig), 3)}
        ok = bool(np.isfinite(p).all() and abs(p.min()) < 1e-9 and abs(p.max() - 1) < 1e-6
                  and n_distinct >= len(p) / 2)
        checks.append(check(f"{layer}: PGI finite, spans [0, 1], ≥ half distinct", n_distinct, len(p) // 2, ok=ok))

    plot(variants, out / "fig11_pgi.png")
    write_gate(out, "step6_pgi", args, checks, info=info)


if __name__ == "__main__":
    main()
