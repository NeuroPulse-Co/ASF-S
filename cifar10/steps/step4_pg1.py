"""
Step 4 – PG1 of the output layer for every (class, correct|incorrect) [paper §4 Eq. 5–8, Fig. 4].

Reads W from step 3 and computes g = PG1 with config.PG_MODE ("paper"), and also with
"legacy" for comparison.

Gate (the failure modes found on CIFAR-100, run 01):
  1. every PG1 is finite and unit length ("paper" mode)
  2. no PG1 is degenerate: ≥ 3 distinct values (the S = 1 collapse gave 1–2 values)
Info: λ1 − λ2 (≈ 0 when W splits into disconnected pieces), λ2 − λ3 (≈ 0 makes v2
unstable), participation ratio PR = 1/Σg⁴ (1 = PG1 sits on one unit, N = spread over all),
|cos| between paper and legacy PG1, W connectivity (from step 3).
Outputs: pg1.npz (g, g_legacy, eigenvalues), fig4_pg1.png, gate.json.
"""

import numpy as np

from common import C, check, parse_args, run_dir, seed_everything, step_dir, write_gate
from asfs.afc import connectivity
from asfs.pg import pg1

SPLITS = ("correct", "incorrect")
AUTO, DOG, TRUCK = 1, 5, 9


def plot(g, path, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(classes), figsize=(3 * len(classes), 4.8), sharey=True, squeeze=False)
    units = np.arange(g.shape[-1])
    for col, c in enumerate(classes):
        for row, split in enumerate(SPLITS):
            a = ax[row, col]
            a.bar(units, g[c, row], color="tab:blue" if row == 0 else "tab:red")
            a.axhline(0, color="k", lw=0.5)
            a.set_title(f"{C.CLASSES[c]} – {split}", fontsize=9)
            a.set_xticks(units); a.set_xticklabels([n[:4] for n in C.CLASSES], rotation=90, fontsize=7)
    fig.suptitle(f"PG1 of the output layer ({C.PG_MODE} mode) – cf. paper Fig. 4", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    args = parse_args("step4_pg1")
    seed_everything(args.seed)
    out = step_dir(args, "step4_pg1")
    W = np.load(run_dir(args) / "step3_afc" / "afc.npz")["W"]          # (classes, 2, N, N)
    n_cls, _, n, _ = W.shape

    g = np.zeros((n_cls, 2, n)); g_leg = np.zeros_like(g); lam = np.zeros_like(g)
    for c in range(n_cls):
        for s in range(2):
            g[c, s], lam[c, s] = pg1(W[c, s], C.ALPHA, C.PG_MODE)
            g_leg[c, s], _ = pg1(W[c, s], C.ALPHA, "legacy")
    np.savez_compressed(out / "pg1.npz", g=g, g_legacy=g_leg, eigenvalues=lam, mode=C.PG_MODE, alpha=C.ALPHA)

    norms = np.linalg.norm(g, axis=-1)
    distinct = np.array([[len(np.unique(np.round(g[c, s], 6))) for s in range(2)] for c in range(n_cls)])
    gap = lam[..., 1] - lam[..., 2]
    gap12 = lam[..., 0] - lam[..., 1]
    pr = 1.0 / np.sum((g / norms[..., None]) ** 4, -1)
    agree = np.abs(np.sum(g * g_leg, -1)) / (norms * np.linalg.norm(g_leg, axis=-1) + 1e-12)
    conn = np.array([[connectivity(W[c, s]) for s in range(2)] for c in range(n_cls)])

    print(f"{'class/split':22s} {'λ2':>7s} {'λ1−λ2':>7s} {'λ2−λ3':>7s} {'PR':>5s} {'distinct':>8s} "
          f"{'|cos| legacy':>12s} {'W conn':>7s}")
    for c in range(n_cls):
        for s, split in enumerate(SPLITS):
            print(f"{C.CLASSES[c] + '/' + split:22s} {lam[c, s, 1]:7.4f} {gap12[c, s]:7.4f} {gap[c, s]:7.4f} "
                  f"{pr[c, s]:5.2f} {distinct[c, s]:8d} {agree[c, s]:12.3f} {conn[c, s]:7.3f}")
    plot(g, out / "fig4_pg1.png", [TRUCK, AUTO, DOG])

    checks = [
        check("all PG1 finite and unit length", round(float(np.abs(norms - 1).max()), 8), 0.0, tolerance=1e-6,
              ok=bool(np.isfinite(g).all() and np.allclose(norms, 1, atol=1e-6))),
        check("min distinct values per PG1 ≥ 3", int(distinct.min()), 3, ok=distinct.min() >= 3),
    ]
    write_gate(out, "step4_pg1", args, checks, info={
        "mode": C.PG_MODE, "alpha": C.ALPHA,
        "min_gap_12": round(float(gap12.min()), 5), "participation_min": round(float(pr.min()), 2),
        "participation_median": round(float(np.median(pr)), 2),
        "min_spectral_gap": round(float(gap.min()), 5), "median_spectral_gap": round(float(np.median(gap)), 5),
        "legacy_abs_cos_min": round(float(agree.min()), 3), "legacy_abs_cos_median": round(float(np.median(agree)), 3),
        "W_connectivity_min": round(float(conn.min()), 3), "W_connectivity_median": round(float(np.median(conn)), 3)})


if __name__ == "__main__":
    main()
