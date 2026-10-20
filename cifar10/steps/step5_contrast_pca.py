"""
Step 5 – PG1 contrast across classes and its PCA [paper §4 Eq. 9–11, §C.3, Fig. 5].

Δg_c = g_corr − g_incorr per class (sign-aligned, see asfs/contrast.py), stacked into ΔG
(10 classes × 10 output units), then PCA with K = 4.

Gate [paper §C.3: "the top 4 components that accounted for ~90% of the variation"]:
  top-4 explained variance ≥ 0.85. One-sided, as in step 1: more than 90% still means
  four components suffice; less would contradict the paper's choice of K.
Info: the full explained-variance curve, the classes whose g_incorr needed a sign flip,
and the top-K share without alignment and for legacy PG1.
Outputs: contrast.npz (dG, components, explained, mean, flips), fig5_contrast.png, gate.json.
"""

import numpy as np

from common import C, check, parse_args, run_dir, seed_everything, step_dir, write_gate
from asfs.contrast import contrast_matrix, pca

GATE_MIN = 0.85


def plot(dG, comps, ratio, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    units = np.arange(dG.shape[1])
    fig = plt.figure(figsize=(14, 6.5))
    for c in range(len(dG)):
        a = fig.add_subplot(3, 5, c + 1)
        a.bar(units, dG[c], color="tab:purple")
        a.axhline(0, color="k", lw=0.5)
        a.set_title(f"Δg {C.CLASSES[c]}", fontsize=9); a.set_xticks([])
    a = fig.add_subplot(3, 5, (11, 12))
    a.plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio), "o-")
    a.axhline(0.9, color="gray", ls="--", lw=0.8); a.axvline(C.PCA_K, color="gray", ls=":", lw=0.8)
    a.set_xlabel("components"); a.set_ylabel("cumulative explained"); a.set_ylim(0, 1.02)
    a = fig.add_subplot(3, 5, (13, 15))
    for k, u in enumerate(comps):
        a.plot(units, u, "o-", label=f"u{k + 1} ({ratio[k]:.0%})")
    a.set_xticks(units); a.set_xticklabels([n[:4] for n in C.CLASSES], rotation=90, fontsize=7)
    a.axhline(0, color="k", lw=0.5); a.legend(fontsize=7)
    fig.suptitle("PG1 contrast per class (Eq. 9) and its PCA (Eq. 11) – cf. paper Fig. 5", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    args = parse_args("step5_contrast_pca")
    seed_everything(args.seed)
    out = step_dir(args, "step5_contrast_pca")
    p = np.load(run_dir(args) / "step4_pg1" / "pg1.npz")
    g, g_leg = p["g"], p["g_legacy"]                                   # (classes, 2, N)
    K = C.PCA_K

    dG, flips = contrast_matrix(g[:, 0], g[:, 1], align=True)
    comps, ratio, mean = pca(dG, K)
    top = float(ratio[:K].sum())
    np.savez_compressed(out / "contrast.npz", dG=dG, components=comps, explained=ratio, mean=mean,
                        flips=flips, K=K)

    # same pipeline without the sign alignment, and on legacy PG1 (original: no alignment)
    _, ratio_noalign, _ = pca(contrast_matrix(g[:, 0], g[:, 1], align=False)[0], K)
    _, ratio_leg, _ = pca(contrast_matrix(g_leg[:, 0], g_leg[:, 1], align=False)[0], K)

    print(f"sign flips needed: {[C.CLASSES[c] for c in np.flatnonzero(flips)] or 'none'}")
    print(f"cumulative explained variance: {np.cumsum(ratio).round(3)}")
    print(f"top-{K}: {top:.3f}  (no alignment {ratio_noalign[:K].sum():.3f}, "
          f"legacy PG1 {ratio_leg[:K].sum():.3f})")
    for k, u in enumerate(comps):
        top_units = np.argsort(-np.abs(u))[:3]
        print(f"  u{k + 1} ({ratio[k]:.1%}): largest on " +
              ", ".join(f"{C.CLASSES[i]} {u[i]:+.2f}" for i in top_units))
    plot(dG, comps, ratio, out / "fig5_contrast.png")

    checks = [check(f"top-{K} explained variance ≥ {GATE_MIN}", round(top, 4), 0.90,
                    tolerance=0.90 - GATE_MIN, ok=top >= GATE_MIN)]
    write_gate(out, "step5_contrast_pca", args, checks, info={
        "K": K, "explained": [round(float(r), 4) for r in ratio],
        "flipped_classes": [C.CLASSES[c] for c in np.flatnonzero(flips)],
        "top_K_no_alignment": round(float(ratio_noalign[:K].sum()), 4),
        "top_K_legacy_pg1": round(float(ratio_leg[:K].sum()), 4)})


if __name__ == "__main__":
    main()
