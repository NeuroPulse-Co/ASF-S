"""
Step 3 – AFC matrices W, Eq. 1–3, for every (class, correct|incorrect) [paper §3, Fig. 3].

For each group, S = 50 images are drawn (seeded) from the step-2 activations. Their
output-layer activations give X^(ℓ) (10 units × S), then A (cosine) and W (Gaussian kernel).
The drawn indices are saved, so step 4 uses exactly the same samples.

Gate [paper Fig. 3, §5.2: "classes that are semantically similar demonstrated similar
looking maps", e.g. truck and automobile vs dog], on the correct-split maps:
  1. d(truck, automobile) < d(truck, dog) and < d(automobile, dog)
  2. mean distance within {vehicles} and within {animals} < mean distance between them
     (the paper's own grouping, App. C.3 Figs. 20–21)
d = Frobenius distance between W matrices.
Info: connectivity of W (σ = 0.1 can make W nearly diagonal, which would leave PG1
meaningless) and the same checks on A, which does not depend on σ.
Outputs: afc.npz (A, W, idx), fig3_afc.png, fig3_all_classes.png, gate.json.
"""

import itertools

import numpy as np

from common import C, check, parse_args, run_dir, seed_everything, step_dir, write_gate
from asfs.afc import afc, connectivity

SPLITS = ("correct", "incorrect")
VEHICLES = [0, 1, 8, 9]                       # airplane, automobile, ship, truck
ANIMALS = [2, 3, 4, 5, 6, 7]                  # bird, cat, deer, dog, frog, horse
AUTO, DOG, TRUCK = 1, 5, 9


def draw(labels, correct, c, split, S, rng):
    pool = np.flatnonzero((labels == c) & (correct if split == "correct" else ~correct))
    return np.sort(rng.permutation(pool)[:S])


def fro(M, a, b):
    return float(np.linalg.norm(M[a] - M[b]))


def similarity_checks(M):
    """M: (classes, N, N) correct-split maps → (pairwise check, grouping check, numbers)."""
    d_ta, d_td, d_ad = fro(M, TRUCK, AUTO), fro(M, TRUCK, DOG), fro(M, AUTO, DOG)
    within = [fro(M, a, b) for g in (VEHICLES, ANIMALS) for a, b in itertools.combinations(g, 2)]
    between = [fro(M, a, b) for a in VEHICLES for b in ANIMALS]
    nums = dict(truck_auto=round(d_ta, 4), truck_dog=round(d_td, 4), auto_dog=round(d_ad, 4),
                within=round(float(np.mean(within)), 4), between=round(float(np.mean(between)), 4))
    return d_ta < min(d_td, d_ad), np.mean(within) < np.mean(between), nums


def plot(W, path, classes, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(classes), figsize=(2.6 * len(classes), 5.4), squeeze=False)
    for col, c in enumerate(classes):
        for row, split in enumerate(SPLITS):
            a = ax[row, col]
            im = a.imshow(W[c, row], vmin=0, vmax=1, cmap="viridis")
            a.set_title(f"{C.CLASSES[c]} – {split}", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    args = parse_args("step3_afc")
    seed_everything(args.seed)
    out = step_dir(args, "step3_afc")
    d = np.load(run_dir(args) / "step2_activations" / "activations.npz")
    acts, labels, correct = d["acts"], d["labels"], d["correct"]
    rng = np.random.default_rng(args.seed)

    n, S = acts.shape[1], C.S_SAMPLES
    A = np.zeros((C.NUM_CLASSES, 2, n, n)); W = np.zeros_like(A)
    idx = np.zeros((C.NUM_CLASSES, 2, S), dtype=np.int64)
    for c in range(C.NUM_CLASSES):
        for s, split in enumerate(SPLITS):
            idx[c, s] = draw(labels, correct, c, split, S, rng)
            A[c, s], W[c, s] = afc(acts[idx[c, s]], C.SIGMA, C.EPS)
    np.savez_compressed(out / "afc.npz", A=A, W=W, idx=idx, sigma=C.SIGMA, eps=C.EPS)

    conn = {f"{C.CLASSES[c]}/{split}": round(connectivity(W[c, s]), 3)
                                 for c in range(C.NUM_CLASSES) for s, split in enumerate(SPLITS)}
    print("W connectivity (fraction of off-diagonal entries > 0.01):")
    for k, v in conn.items():
        print(f"  {k:22s} {v:.3f}")

    pair_W, group_W, nums_W = similarity_checks(W[:, 0])
    pair_A, group_A, nums_A = similarity_checks(A[:, 0])
    print("W distances:", nums_W)
    print("A distances:", nums_A)

    plot(W, out / "fig3_afc.png", [TRUCK, AUTO, DOG], f"AFC W (σ={C.SIGMA}), S={S} – cf. paper Fig. 3")
    plot(W, out / "fig3_all_classes.png", list(range(C.NUM_CLASSES)), f"AFC W (σ={C.SIGMA}), all classes")

    checks = [
        check("W: d(truck,auto) < d(truck,dog), d(auto,dog)", nums_W["truck_auto"],
              min(nums_W["truck_dog"], nums_W["auto_dog"]), ok=pair_W),
        check("W: mean within-group < between (vehicles / animals)", nums_W["within"],
              nums_W["between"], ok=group_W),
    ]
    write_gate(out, "step3_afc", args, checks, info={
        "S": S, "sigma": C.SIGMA, "units": n,
        "W_connectivity_min": min(conn.values()), "W_connectivity_mean": round(float(np.mean(list(conn.values()))), 3),
        "A_checks": {"pair": bool(pair_A), "group": bool(group_A), **nums_A}, "W_distances": nums_W})


if __name__ == "__main__":
    main()
