"""
diag_sigma – which σ (Eq. 3) gives a well-defined, stable PG1? The paper does not state σ.

At σ = 0.1 (the original scripts' value) the 10×10 output-layer W falls apart into
disconnected pieces. K then has several eigenvalues equal to 1, so "v2" is an arbitrary
vector in a repeated eigenspace (step 4, run01). This sweeps σ and reports, per value:

  conn      median fraction of off-diagonal W > 0.01
  gap12     min over the 20 groups of λ1 − λ2  (≈ 0: disconnected graph)
  gap23     min over the 20 groups of λ2 − λ3  (≈ 0: v2 not unique)
  PR        median participation ratio of PG1 (1 = one unit, 10 = all)
  stab      median |cos| between PG1 from two independent S-sample draws of the same group,
            averaged over R draw pairs. 1 = PG1 is a property of the class, ~0.3 = noise.
  fig3      step 3's two similarity checks still pass on W

"median" = the median of all off-diagonal (1 − A) over the 20 groups (pooled), a standard
bandwidth rule for Gaussian kernels.
Reads step 2's activations; no GPU needed. Output: diag_sigma.json in the run directory.
"""

import json

import numpy as np

from common import C, parse_args, run_dir
from asfs.afc import afc, connectivity, cosine_similarity, activation_matrix
from asfs.pg import pg1
from step3_afc import draw, similarity_checks

SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def groups(labels, correct, S, rng):
    return [[draw(labels, correct, c, split, S, rng) for split in ("correct", "incorrect")]
            for c in range(C.NUM_CLASSES)]


def evaluate(acts, draws_a, draws_b_list, sigma):
    gap12, gap23, pr, conn, stab = [], [], [], [], []
    W0 = np.zeros((C.NUM_CLASSES, acts.shape[1], acts.shape[1]))
    for c in range(C.NUM_CLASSES):
        for s in range(2):
            _, W = afc(acts[draws_a[c][s]], sigma, C.EPS)
            g, lam = pg1(W, C.ALPHA, "paper")
            if s == 0:
                W0[c] = W
            gap12.append(lam[0] - lam[1]); gap23.append(lam[1] - lam[2])
            pr.append(1 / np.sum(g ** 4)); conn.append(connectivity(W))
            cos = []
            for draws_b in draws_b_list:
                gb, _ = pg1(afc(acts[draws_b[c][s]], sigma, C.EPS)[1], C.ALPHA, "paper")
                cos.append(abs(float(g @ gb)))
            stab.append(np.mean(cos))
    pair, group, _ = similarity_checks(W0)
    return dict(conn=float(np.median(conn)), gap12=float(np.min(gap12)), gap23=float(np.min(gap23)),
                PR=float(np.median(pr)), stab=float(np.median(stab)), stab_min=float(np.min(stab)),
                fig3=bool(pair and group))


def main():
    args = parse_args("diag_sigma", lambda ap: ap.add_argument(
        "--redraws", type=int, default=5, help="independent second draws per group for stability"))
    d = np.load(run_dir(args) / "step2_activations" / "activations.npz")
    acts, labels, correct = d["acts"], d["labels"], d["correct"]
    rng = np.random.default_rng(args.seed)
    draws_a = groups(labels, correct, C.S_SAMPLES, rng)
    draws_b = [groups(labels, correct, C.S_SAMPLES, rng) for _ in range(args.redraws)]

    one_minus_a = np.concatenate([
        (1 - cosine_similarity(activation_matrix(acts[draws_a[c][s]])))[~np.eye(acts.shape[1], dtype=bool)]
        for c in range(C.NUM_CLASSES) for s in range(2)])
    sigma_med = float(np.median(one_minus_a))
    print(f"off-diagonal 1 − A: quartiles {np.percentile(one_minus_a, [25, 50, 75]).round(3)} "
          f"→ median rule σ = {sigma_med:.3f}")

    report = {}
    print(f"\n{'σ':>8s} {'conn':>6s} {'gap12':>8s} {'gap23':>8s} {'PR':>5s} {'stab':>6s} {'stab_min':>8s} {'fig3':>5s}")
    for name, sigma in [(str(s), s) for s in SIGMAS] + [(f"median", sigma_med)]:
        r = evaluate(acts, draws_a, draws_b, sigma)
        report[name] = {"sigma": sigma, **r}
        print(f"{name:>8s} {r['conn']:6.3f} {r['gap12']:8.5f} {r['gap23']:8.5f} {r['PR']:5.2f} "
              f"{r['stab']:6.3f} {r['stab_min']:8.3f} {'yes' if r['fig3'] else 'no':>5s}")

    out = run_dir(args) / "diag_sigma.json"
    out.write_text(json.dumps({"sigma_median_rule": sigma_med, "sweep": report}, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
