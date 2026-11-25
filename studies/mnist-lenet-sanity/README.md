# ASF-S on MNIST with LeNet-300-100

A self-contained, CPU-only re-implementation of ASF-S (Artificial Structure Function
Search) for the paper's MNIST experiment. The whole pipeline runs on a laptop in about
30 minutes, most of it fine-tuning in stage 4.

## Quick start

Run from this folder with the repo venv:

```bash
python test_sanity.py      # 9 correctness checks (~10 s)
python extract.py          # stage 1: activations              (~5 s, downloads MNIST once)
python pg.py --plot        # stage 2: AFC maps + PG1           (~5 s)
python importance.py       # stage 3: PCA + PGI + baselines    (~2 s)
python prune.py            # stage 4: prune fc1 + fine-tune    (~30 min)
python prune.py --mode threshold --ft head   # paper-style threshold sweep
```

The model is the paper's own checkpoint, restored from git history
(`git show 6b81a13:CNN/models/lenet300_100_best.pth`). It reproduces paper Table 4
exactly: 98.51% test accuracy, 266,610 parameters and 532,400 FLOPs.

## The pipeline, stage by stage

| Stage | File | Paper | Produces |
|---|---|---|---|
| 1 | `extract.py` | Eq. 1 | `outputs/activations.npz`: every analysis-split sample's fc1/fc2/fc3 activations, labels and correct/incorrect flags |
| 2 | `pg.py` | Eq. 2–8 | `outputs/pg/<variant>_<layer>.npz`: PG1 of the correct and incorrect groups for each class, and their contrast |
| 3 | `importance.py` | Eq. 9–13 | `outputs/importance/<method>.npy`: one score in [0, 1] per fc1 neuron |
| 4 | `prune.py` | Sec. 5 | `results/prune_<mode>.csv`: accuracy, ECE, params and FLOPs for every method, sparsity and fine-tuning mode |

### Data roles

| Split | Used for |
|---|---|
| MNIST train (60k) | fine-tuning after pruning |
| test "analysis" half (5k) | building PG1 (it needs misclassified samples); early stopping |
| test "eval" half (5k) | **reporting only** |

The PG1 data comes from the test set because the checkpoint is 99.97% accurate on
the training set. The original extraction (first 1000 train images per class) found
exactly one misclassified image, in class 2. That is why every "incorrect" panel in
the paper's Fig. 14 except class 2 is empty. The analysis half has 3–12 misclassified
images per class.

## Where this differs from the original code, and why

| # | Original (`min_scripts/`, `cifar100/`) | Here | Why |
|---|---|---|---|
| 1 | PG1 from **one** sample (`break` after the first file) | up to S=50 samples | With S=1, PG1 reduces to the sign pattern of one image: 2 distinct values. `pg.py` prints this for the `as_coded` variant. |
| 2 | hooks on `nn.Linear` record **pre-ReLU** values | post-ReLU | The paper's argument for cosine similarity assumes non-negative activations. |
| 3 | σ = 0.1 | σ = 0.5 | At 0.1 the fc2 unit graph splits into 15–22 disconnected pieces, so the eigenvalue 1 repeats and PG1 is not unique. 0.5 is the smallest σ that connects the graph for both fc2 and fc3 (see below). |
| 4 | dead units kept in the graph | excluded, PG1 = 0 | An all-zero unit becomes an isolated node with eigenvalue ~1 and can take over PG1. |
| 5 | eigenvector sign left arbitrary | fixed convention, g_incorr aligned to g_corr | v and −v are the same eigenvector; for unit vectors g_c − g_i is orthogonal to g_c + g_i, so an arbitrary sign makes the contrast direction a coin flip. |
| 6 | `eigh` on the row-normalised (non-symmetric) M | `eigh` on the symmetric K of Eq. 6 | Eq. 7 is an eigendecomposition of K. |
| 7 | Eq. 12 truncated to the shortest vector | dimensionally consistent scores (`asf_fc2`, `asf_fc3`) | For fc1 the truncation reads pixels 0–99 (image rows 0–3, background). The pixels read under the paper's own output-layer choice (0–9) are blank in 100% of images. |
| 8 | PCA `n_components` capped only by the number of units | also capped by n_classes − 1 | PCA centres the data first; this also stops the single-class crash. |
| 9 | FC weights copied by prefix (`cifar100/rebuild_fc`) | exact row and column selection | The pruned net must equal the original with those neurons silenced (tested). |
| 10 | comparison by threshold | comparison at matched sparsity (top-k) | A PGI threshold of 0.7 is not 70% sparsity. |

`asf_as_coded` keeps every original behaviour (1, 2, 3, 5, 6, 7) so the paper's
procedure can be compared side by side.

### Choosing σ

The number of eigenvalues equal to 1 is the number of disconnected pieces of the
graph, and PG1 is unique only if that number is 1. The gap λ2 − λ3 measures how
stable PG1 is under resampling. Scan on 50 correct samples per class:

| σ | fc2 pieces | fc2 gap | fc3 pieces | fc3 gap |
|---|---|---|---|---|
| 0.1 | 15–22 | 0 | 2–4 | ≤ 0.11 |
| 0.3 | 1 | 0.01–0.08 | 2 | — |
| **0.5** | **1** | 0.03–0.08 | **1** | **0.64–0.86** |
| 1.0 | 1 | ~0.01 | 1 | ~0.3 |

`pg.py` prints the pieces and the gap for every class, and warns when the graph is
disconnected.

## Importance methods (stage 3)

For fc1 neuron j (W1 row j = incoming weights, W2 column j = outgoing weights):

| Method | Score before Eq. 13 averaging | PG layer |
|---|---|---|
| `asf_as_coded` | ⟨W1[j, :100], u_k[:100] ⊙ w̄3[:100]⟩, with \|mean\| over k | fc2, S=1 |
| `asf_fc2` | ⟨W2[:, j], u_k ⊙ w̄3⟩, where w̄3 = mean row of W3 | fc2, S≤50 |
| `asf_fc3` | ⟨W3 · W2[:, j], u_k⟩ (linear route from neuron j to the logits) | fc3 (output layer, as in the paper) |
| `l1`, `l2` | ‖W1[j, :]‖₁, ‖W1[j, :]‖₂ | — |
| `random_s0..4` | uniform noise, 5 seeds | — |

## Findings so far

- **The PG1 contrast of the output layer is meaningful.** For 9 of 10 classes, the
  most negative contrast entries fall on the classes the model actually confuses that
  digit with (4→9, 7→2, 9→4, 2→7/8, …).
- **`asf_fc3`'s top 4 PCA components explain 92% of the variance**, consistent with
  the paper's "~90%" (App. C.3). The fc2-based variants reach 73%.
- **ASF-S rankings correlate 0.85–0.93 (Spearman) with plain L1/L2 magnitude.** Any
  Eq. 12-style score is a dot product with the neuron's weights, so it grows with the
  weight norm. The fair baseline for ASF-S is therefore magnitude pruning, not random.
### Pruning results (`results/prune_ratio.csv`, `results/accuracy_vs_sparsity.png`)

Eval-split accuracy (%) with fc1 pruned to matched sparsity. The unpruned model scores
98.34. With 5,000 eval images the measurement noise at ~97% is about ±0.24 points, so
differences below ~0.7 points should not be read as real.

| fc1 pruned | setting | asf_fc3 | asf_fc2 | as_coded | L1 | L2 | random (5 seeds) |
|---|---|---|---|---|---|---|---|
| 70% | no FT | 94.10 | 93.50 | 92.82 | 97.22 | 97.44 | 75.7 ± 4.0 |
| 70% | fc3 only | 97.40 | 97.10 | 96.66 | 97.86 | 97.82 | 92.1 ± 1.1 |
| 70% | full FT | 98.14 | 98.32 | 97.84 | 98.10 | 98.16 | 97.8 ± 0.3 |
| 90% | no FT | 57.53 | 61.19 | 60.77 | 72.03 | 68.13 | 31.7 ± 4.3 |
| 90% | fc3 only | 84.24 | 89.10 | 86.40 | 90.48 | 91.20 | 68.4 ± 3.7 |
| 90% | full FT | 96.44 | 97.36 | 97.36 | 96.86 | 97.00 | 96.5 ± 0.3 |

- **ASF-S beats random by a wide margin** whenever the pruned layers are frozen.
- **ASF-S does not beat L1/L2 magnitude pruning.** Without fine-tuning, or retraining
  fc3 only, magnitude pruning is equal or better at every sparsity, by 3–8 points at
  70–90% without fine-tuning.
- **The paper's Table 1 claim is not reproduced.** At 70% sparsity with fc3-only
  retraining, the drop is 0.9 points (asf_fc3) or 1.7 points (as_coded, the paper's
  procedure), not the reported 0.
- **Full fine-tuning cannot tell the methods apart on MNIST.** Every method, random
  included, recovers to within ~1 point up to 80% sparsity.
- **The PG1 contribution looks small.** The as_coded score reads only background
  pixels, yet performs close to the consistent variants. That points to the
  weight-norm factor in Eq. 12, not PG1, doing most of the work.

Caveats: one checkpoint, one fine-tuning seed, only fc1 pruned, σ = 0.5.
`asf_fc2` and `asf_fc3` are dimensionally consistent readings of Eq. 12, not the
paper's literal formula (which cannot be computed as written).

### Comparison with the paper's LeNet claims

The threshold and 95% runs below come from:
`prune.py --mode threshold --thresholds 0.2 0.8 --methods asf_as_coded asf_fc2 asf_fc3`
and `prune.py --ratios 0.95 --methods asf_as_coded asf_fc2 asf_fc3 l2`.

| Paper claim | Paper | Here | Reproduced? |
|---|---|---|---|
| Table 4 baseline | 98.51%, 267K params, 0.53 MFLOPs | 98.51%, 266,610 params, 532,400 FLOPs | yes, exactly |
| App. C.3: 4 PCA components ≈ 90% variance | ~90% | 92% (asf_fc3) | yes |
| Table 1: 70% sparsity, 0 accuracy drop | 0 | full FT −0.02 to −0.50; fc3-only −0.94 to −1.68 | only with full fine-tuning |
| Table 5: ASF-S at ~95% params removed | −2.89 | −2.64 to −3.62 (full FT, 94.6%) | yes, close |
| Table 5 / §B.2: ASF-S beats L2 | (paper's own LeNet row: L2 −1.80 vs ASF-S −2.89) | L2 ≥ ASF-S at every sparsity | no |
| §B.1: threshold 0.2 → ~62% params removed | 62% | 45–49% (consistent variants), 84% (as_coded) | roughly |
| §B.1: threshold 0.8 → 97% params, accuracy within 4% | within 4% | 95.6–98.9% params; accuracy −4.3 (asf_fc3, full FT) to −73 | no (accuracy) |
| Abstract: baseline accuracy without retraining pruned layers | yes | best fc3-only drop 0.94; L1 0.48 | no, strictly |

The script behind the paper's MNIST numbers is not in the repo, so its fine-tuning
mode and evaluation protocol are unknown. The pattern above is consistent with the
paper's LeNet numbers coming from full fine-tuning.

### Suggested next experiments

1. A magnitude-free ASF-S score: the cosine of W2[:, j] (or W3·W2[:, j]) with u_k
   instead of the dot product. This isolates PG1's own contribution.
2. 3–5 seeds for fine-tuning and for the data split, to get error bars.
3. σ sensitivity (0.3 / 0.5 / 1.0), and `prune.py --mode threshold` to reproduce the
   paper's threshold figures (Fig. 6, App. B.1).
