"""
importance.py - STAGE 3: turn the PG1 contrasts into one importance score per fc1
neuron (paper Eq. 9-13), plus baseline scores to compare against.

Paper recipe
------------
  Eq. 9-10  dG = stack_c (g_c^corr - g_c^incorr)            (C x units), built in pg.py
  Eq. 11    PCA(dG) -> top-K components u_1..u_K             class-shared directions
  Eq. 12    s_{i,k} = < w_i , u_k (.) w_bar >                 (.) = element-wise product
  Eq. 13    PGI_i = minmax( (1/K) sum_k |s_{i,k}| )           low PGI -> prune first

The dimension problem with Eq. 12, and how each method handles it
-----------------------------------------------------------------
For a fc1 neuron, the paper's w_i is its INCOMING weight vector: 784 entries, indexed
by input pixel. u_k is indexed by the units of the PG layer (100 for fc2, 10 for fc3),
and w_bar (the mean output-layer weight row) by fc2 units. The three vectors live in
different spaces, so a dot product between them is undefined.

  asf_as_coded  What the original code does: truncate all three vectors to the shortest
                length and dot them anyway (min_scripts/pg_pruning_conv2.py:95-107).
                For LeNet that means pixel k is paired with fc2 unit k, for k < 100:
                the score reads only image rows 0-3, which are blank background in
                MNIST. Also uses |mean_k s| rather than Eq. 13's mean_k |s|.

  asf_fc2       Dimensionally consistent reading of Eq. 12 with the PG layer = fc2:
                use the neuron's OUTGOING weights W2[:, j] (indexed by fc2 units), so
                    s_{j,k} = < W2[:, j] , u_k (.) w_bar3 >,   w_bar3 = mean_c W3[c, :]
                Every vector is now indexed by fc2 units. Interpretation: how strongly
                neuron j feeds the fc2 units that carry the correct-vs-incorrect axis.

  asf_fc3       PG layer = the OUTPUT layer, as the paper's text states. Neuron j's
                linear route to the logits is v_j = W3 @ W2[:, j]   (10 entries), so
                    s_{j,k} = < W3 @ W2[:, j] , u_k >
                (fc2's ReLU is ignored, i.e. a first-order approximation). There is no
                layer after fc3, so there is no w_bar factor here.

Baselines: l1 / l2 = norm of the neuron's incoming weights (classic magnitude pruning,
the "L1/L2" rows of paper Table 5), random = uniform noise, one set per seed.

Output: outputs/importance/<method>.npy, shape (300,), values in [0, 1].

Run:  python importance.py
"""

import argparse
import os

import numpy as np
from sklearn.decomposition import PCA

import config
from models import load_lenet


def minmax(x):
    """Eq. 13's normaliser N(.): rescale to [0, 1] within the layer."""
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def pca_components(pg_file, k):
    """Eq. 11: PCA over the valid rows of the contrast matrix dG."""
    r = np.load(pg_file)
    dG = r["delta"][r["valid"]]
    # sklearn PCA centres the rows first, so C rows give at most C-1 meaningful
    # components. The original code only capped k by the number of units, which is why
    # it would crash with the single class the train-set extraction produced.
    k = min(k, dG.shape[0] - 1, dG.shape[1])
    pca = PCA(n_components=k).fit(dG)
    return pca.components_, pca.explained_variance_ratio_


def score_as_coded(W1, W3, comps):
    """Replica of compute_importance() in min_scripts/pg_pruning_conv2.py:95-107 and in
    git 6b81a13:CNN/cifar100_original_pipeline/pg_pruning_cifar100.py (lenet branch):
    incoming fc1 weights, downstream = fc3 weights, prefix truncation, |mean|."""
    avg_fc = W3.mean(axis=0)                                  # (100,), indexed by fc2 unit
    scores = []
    for comp in comps:                                        # comp: (100,), fc2 units
        sz = min(W1.shape[1], avg_fc.shape[0], comp.shape[0]) # = 100
        scores.append(W1[:, :sz] @ (comp[:sz] * avg_fc[:sz])) # uses pixels 0..99 only
    return minmax(np.abs(np.mean(scores, axis=0)))            # |mean_k s| (not Eq. 13)


def score_fc2(W2, W3, comps):
    """s_{j,k} = <W2[:, j], u_k (.) w_bar3>. Vectorised: W2.T (300 x 100) @ (100,)."""
    w_bar3 = W3.mean(axis=0)                                  # (100,)
    s = np.stack([W2.T @ (u * w_bar3) for u in comps])        # (K, 300)
    return minmax(np.abs(s).mean(axis=0))                     # Eq. 13: mean_k |s|


def score_fc3(W2, W3, comps):
    """s_{j,k} = <W3 @ W2[:, j], u_k>. (W3 @ W2) is (10 x 300): column j is neuron j's
    linear effect on the 10 logits."""
    route = W3 @ W2                                           # (10, 300)
    s = np.stack([route.T @ u for u in comps])                # (K, 300)
    return minmax(np.abs(s).mean(axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=config.CKPT_PATH)
    ap.add_argument("--k", type=int, default=config.PCA_COMPONENTS)
    ap.add_argument("--random_seeds", type=int, default=5)
    args = ap.parse_args()

    m = load_lenet(args.ckpt)
    W1 = m.fc1.weight.detach().numpy()     # (300, 784) incoming weights of each fc1 neuron
    W2 = m.fc2.weight.detach().numpy()     # (100, 300) column j = outgoing weights of fc1 neuron j
    W3 = m.fc3.weight.detach().numpy()     # (10, 100)

    scores = {}
    for method, pg_name, fn in [("asf_as_coded", "as_coded_fc2", lambda c: score_as_coded(W1, W3, c)),
                                ("asf_fc2", "paper_fc2", lambda c: score_fc2(W2, W3, c)),
                                ("asf_fc3", "paper_fc3", lambda c: score_fc3(W2, W3, c))]:
        comps, evr = pca_components(os.path.join(config.PG_DIR, f"{pg_name}.npz"), args.k)
        print(f"{method:13s} PG={pg_name:12s} PCA k={len(comps)}  explained variance "
              f"{np.round(evr, 3)} (sum {evr.sum():.2f})")
        scores[method] = fn(comps)

    scores["l1"] = minmax(np.abs(W1).sum(axis=1))
    scores["l2"] = minmax(np.linalg.norm(W1, axis=1))
    for s in range(args.random_seeds):
        scores[f"random_s{s}"] = np.random.default_rng(1000 + s).random(W1.shape[0])

    for name, v in scores.items():
        np.save(os.path.join(config.IMPORTANCE_DIR, f"{name}.npy"), v)

    # ── How different are the rankings? Spearman rank correlation between methods ──
    # (1 = identical ordering of neurons, 0 = unrelated). Rank-based because pruning
    # only depends on the ORDER of the scores.
    names = ["asf_as_coded", "asf_fc2", "asf_fc3", "l1", "l2", "random_s0"]
    ranks = {n: np.argsort(np.argsort(scores[n])) for n in names}
    print("\nSpearman rank correlation between fc1 neuron rankings:")
    print(" " * 14 + "".join(f"{n[:10]:>11s}" for n in names))
    for a in names:
        print(f"{a:14s}" + "".join(f"{np.corrcoef(ranks[a], ranks[b])[0, 1]:11.2f}" for b in names))
    print(f"\nSaved {len(scores)} score files -> {config.IMPORTANCE_DIR}")


if __name__ == "__main__":
    main()
