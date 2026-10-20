# ASF-S CIFAR-10 re-implementation log

A running record of the decisions, runs, gate results and findings for `cifar10/`. The newest entry is first.
`README.md` holds the current status table. This log holds the reasoning and evidence behind it.

Each run records the host, commit, command and result. Machine: `intellisense08-EWISPro9900G`, RTX 2080,
driver 535, torch 2.5.1+cu121.

---

# 2026-10-20 — Step 5: top-4 PCA components explain 92.7%, against the paper's ~90%

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 5` (σ = 0.831 PG1, sign alignment on, K = 4) |
| Result | **PASS**. Top-4 explained variance **0.927**, against the paper's "~90%" (§C.3) |

Cumulative explained variance: 0.448, 0.652, 0.809, **0.927**, 0.977, 0.992, 0.997, 0.999, 1, 1.

| Component | Share | Largest loadings (output units) |
|---|---|---|
| u1 | 44.8% | bird +0.54, automobile −0.43, cat +0.41 |
| u2 | 20.4% | cat +0.60, deer −0.44, airplane −0.39 |
| u3 | 15.7% | horse +0.61, bird −0.52, dog +0.31 |
| u4 | 11.8% | airplane +0.52, dog −0.47, frog +0.38 |

## Notes

- **This is the first paper number the PG pipeline matches directly.**
- **No class needed a sign flip.** The canonical sign from step 4 (largest-|·| entry positive) already keeps each
  class's g_corr and g_incorr pointing the same way, so the result is the same with and without alignment (0.927).
- **Legacy PG1 gives 0.962,** more concentrated, as expected from its λ^½ scaling, which exaggerates
  differences between classes.
- **u1 roughly separates animals (bird, cat +) from vehicles (automobile −).** It is the same split that
  organises the AFC maps in step 3. The later components contrast specific confusable pairs.
- Cosmetic fix: the gate tolerance is now rounded (it printed `0.050000000000000044`).

## Step 6 prepared: PGI (`asfs/pgi.py`, `steps/step6_pgi.py`)

- **The problem with Eq. 12.** u_k ∈ R¹⁰ lives on the output units, so u_k ⊙ w̄^(f) needs w̄^(f) ∈ R¹⁰, and w_i
  must then be in R¹⁰ too. A conv filter's flattened weights have 27 values (conv1) or 288 (conv2). The paper
  never says how to map them.
- **`literal` mode** is the original code, verified line-for-line against `min_scripts/pg_pruning_conv2.py` in
  `test_step6.py`:
  - w̄ is the mean over FC *rows*, giving 16,384 values.
  - Everything is cut to the first 10 entries, so for conv1 the score uses only the red-channel kernel and one
    green weight.
  - |mean_k s|.
- **`aligned` mode** (the default):
  - w̄^(f) is the mean over FC *columns*, giving one value per output unit.
  - w_i is replaced by e_i ∈ R¹⁰, filter i's effective linear weight to each output unit. For conv2, e_i is the
    mean of the 256 FC columns its channel feeds, times its BN scale. For conv1, it is propagated back through
    conv2's kernel-summed weights and BN1's scale. ReLU and pooling are ignored.
  - Eq. 13 as written: mean_k |s|.
- **The paper gives no number for this step,** so the gate is a sanity check: finite, spans [0, 1], not
  degenerate. The informative outputs are:
  - per layer, the fraction of filters below each τ, which predicts step 7's sparsity grid
  - Spearman ρ against the filter L1 norm (does PGI add anything over magnitude?)
  - Spearman ρ between aligned and original

---

# 2026-10-20 — Step 4: at σ = 0.1, PG1 is not defined (repeated eigenvalue 1). Gate tightened

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 4` (paper mode, σ = 0.1, α = 0.5) |
| Gate as written | PASS (unit length; ≥ 8 distinct values) |
| Actual result | **invalid**: in all 20 groups, λ1 = λ2 = λ3 = 1.0000 (λ1 − λ2 = λ2 − λ3 = 0) |

| | range over the 20 groups |
|---|---|
| W connectivity (off-diagonal > 0.01) | 0.067 – 0.356 |
| participation ratio | 1.17 – 8.21 |
| \|cos\| paper vs legacy PG1 | 0.000 – 0.992 |

## Diagnosis

- **The graph falls apart.** With σ = 0.1, W_ij = exp(−(1 − A_ij)²/0.02) drops below 0.01 as soon as the cosine
  falls under about 0.7. Most unit pairs among the 10 output logits are below that, so each W splits into several
  disconnected pieces. Every piece gives K an eigenvalue of exactly 1.
- **So "v₂" is not unique.** It is an arbitrary vector in a repeated eigenspace, whatever `eigh` happens to
  return. The erratic paper-vs-legacy agreement (0.000 to 0.992) and the erratic participation ratios show this.
- **Reproduced locally** on synthetic structured activations: σ = 0.1 gives λ = (1, 1, 1); σ = 0.3 gives
  (1, 0.994, 0.888); σ = 1.0 gives (1, 0.476, 0.208).
- **The original scripts were also degenerate.** They used σ = 0.1 with S = 1: every cosine is ±1, so W is two
  disconnected blocks, again with a repeated eigenvalue 1. The paper's PG1, and everything built on it, was
  therefore very likely ill-defined. **This is a major reproduction finding.**

## Changes

- **Step 4 gate** gains two checks, in every group: λ1 − λ2 > 1e-3 (W connected) and λ2 − λ3 > 1e-3 (v₂
  unique). run01 at σ = 0.1 now fails, correctly.
- **`steps/diag_sigma.py`** (`./cifar10/run.sh diag_sigma`) sweeps σ ∈ {0.1, 0.2, 0.3, 0.5, 0.75, 1.0, median
  rule}. For each it reports connectivity, the minimum eigen-gaps, the participation ratio, whether step 3's
  Fig. 3 checks still hold, and **stability**: |cos| between PG1 built from two independent 50-image draws of the
  same group. A PG1 that reflects the class should be stable; noise is not.

## σ sweep (`./cifar10/run.sh diag_sigma`)

Off-diagonal 1 − A quartiles: 0.459 / **0.831** / 1.272, so the median rule gives σ = 0.831.

| σ | conn | min λ1−λ2 | min λ2−λ3 | PR | stability median | stability min | Fig. 3 |
|---|---|---|---|---|---|---|---|
| 0.1 (original) | 0.156 | 0 | 0 | 2.95 | 0.673 | 0.282 | yes |
| 0.2 | 0.344 | 0 | 0.00009 | 2.49 | 0.951 | 0.332 | yes |
| 0.3 | 0.556 | 0.00030 | 0.02751 | 2.56 | 0.981 | 0.812 | yes |
| 0.5 | 0.856 | 0.04961 | 0.17341 | 2.63 | 0.984 | 0.885 | yes |
| 0.75 | 1.000 | 0.31326 | 0.17296 | 3.00 | 0.986 | 0.922 | yes |
| 1.0 | 1.000 | 0.55409 | 0.11611 | 3.39 | 0.988 | 0.934 | yes |
| **median (0.831)** | 1.000 | **0.40532** | **0.15215** | 3.13 | **0.987** | **0.927** | yes |

Stability = |cos| between PG1 built from two independent 50-image draws of the same group, averaged over
5 redraws.

## Decision: σ = median rule

- **The rule fixed in advance** was: the median rule if it meets both gap conditions (> 1e-3) and gives stable
  PG1. It meets both by a wide margin, so `config.SIGMA = "median"`.
- **How it is applied.** Step 3 computes σ from its own cosine matrices and saves it in `afc.npz`, so the rule
  adapts to the `clean` protocol and to Conv6Net. `SIGMA_LEGACY = 0.1` is kept.
- **Robustness:** from σ = 0.5 to 1.0 every metric is flat (stability 0.98–0.99, Fig. 3 holds). The choice is
  not fitted to a particular outcome.
- **Finding:** at a valid σ, PG1 is a stable property of each (class, split) group. Independent 50-image draws
  give nearly the same vector (median |cos| 0.987). This is the first positive evidence for the paper's PG1 idea.
- **At σ = 0.1** even the stability is poor (minimum 0.28), in line with the degenerate eigenspace.

## Next

- [x] Reran step 3 at **σ = 0.8309** (median rule, the same value as the sweep): **PASS**

  | | σ = 0.1 (before) | σ = 0.831 |
  |---|---|---|
  | W connectivity, all 20 groups | 0.067–0.356 | **1.000** |
  | d(truck, automobile) | 1.085 | **0.834** |
  | d(truck, dog), d(automobile, dog) | 2.369, 2.699 | 4.859, 4.852 |
  | within / between (vehicles vs animals) | 1.290 / 2.452 (ratio 0.53) | 3.191 / 4.399 (ratio 0.73) |

  - Every unit pair is now connected.
  - The truck–automobile vs dog separation is stronger: 5.8×, up from 2.2×.
  - The vehicles/animals grouping still holds but has a smaller margin.
  - A is unchanged, as expected.
- [x] Reran step 4 at σ = 0.831: **PASS** on all four checks. min λ1 − λ2 = 0.4053 and min λ2 − λ3 = 0.1521,
  exactly as the sweep predicted. Every PG1 has 10 distinct values.

  | | correct groups | incorrect groups |
  |---|---|---|
  | λ2 | 0.41–0.59 | 0.35–0.55 |
  | participation ratio | 1.84–4.76 (mostly < 3.2) | 3.11–8.45 |
  | \|cos\| paper vs legacy | 0.990–0.995 | 0.993–0.999 |

  - **Paper and legacy PG1 now agree** (|cos| ≥ 0.99, up from 0.00–0.99 at σ = 0.1). The original code's
    differences (row-normalised M, `eigh` on a non-symmetric matrix, λ^½ scaling) barely change the *direction*
    of PG1 once W is connected. **The original method failed because of σ = 0.1 and S = 1, not because of the
    eigensolver.**
  - **Correct-split PG1 is concentrated on a few output units, incorrect-split PG1 is spread out.** When the model
    is right, the true class and its usual confusions dominate the co-activation structure. When it is wrong, more
    units take part. This difference is what the step 5 contrast Δg = g_corr − g_incorr picks up.

---

# 2026-10-20 — Step 3: AFC maps reproduce the paper's Fig. 3 similarity structure

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 3` |
| Setup | S = 50 per (class, split), drawn with seed 42 from step 2. σ = 0.1, ε = 1e-6. 10 output units → W is 10×10 |
| Result | **PASS** on both checks (correct-split maps, Frobenius distance) |

| | W (σ = 0.1) | A (cosine, no kernel) |
|---|---|---|
| d(truck, automobile) | **1.085** | **1.998** |
| d(truck, dog) | 2.369 | 8.532 |
| d(automobile, dog) | 2.699 | 8.573 |
| mean within {vehicles}, {animals} | **1.290** | **5.731** |
| mean between vehicles and animals | 2.452 | 7.710 |

## Notes

- **The paper's §5.2 / Fig. 3 claim holds clearly.** Semantically similar classes have similar maps: truck and
  automobile are 2.2–2.5× closer to each other than to dog. The vehicle/animal grouping (App. C.3) separates
  about 2× as well.
- **The structure is already in A,** before the Gaussian kernel. It is not an artefact of σ.
- **Still to record:** W connectivity, i.e. how much of W survives σ = 0.1 (in `gate.json` → `info`). Step 4
  depends on it: a W close to εI gives meaningless eigenvectors.

## Step 4 prepared: PG1 (`asfs/pg.py`, `steps/step4_pg1.py`)

- **`paper` mode:** `eigh` on the symmetric K = D^-α W D^-α, v₂ unit length, sign fixed so the largest-|·|
  entry is positive.
- **`legacy` mode:** the original computation. Row-normalised M, `eigh` on that non-symmetric matrix, and v₂
  scaled by λ₂^0.5. Computed alongside for comparison.
- **Found while testing locally:** for random activations with σ = 0.1, W ≈ εI and PG1 comes out **one-hot**
  ([1, 0, …, 0]). So step 4 reports, per group:
  - λ1 − λ2 (≈ 0 means a disconnected graph)
  - λ2 − λ3 (≈ 0 means an unstable v₂)
  - the participation ratio 1/Σg⁴ (1 = one-hot, 10 = spread over all units)
  - W connectivity
- **Gate:** every PG1 finite and unit length, with ≥ 3 distinct values. The distinct-values check catches the
  S = 1 collapse and one-hot vectors. The participation ratio shows partial localization.

---

# 2026-10-20 — Step 2: output-layer activations, enough errors in every class

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 2` (paper protocol, run01 model) |
| PG set | 1000 per class from the training set (`train-subset`), no augmentation |
| Result | **PASS**. Smallest (class, split) group: 71 (automobile, incorrect), against S = 50 |

| class | correct | incorrect |
|---|---|---|
| airplane | 816 | 184 |
| automobile | 929 | **71** |
| bird | 676 | 324 |
| cat | 711 | 289 |
| deer | 712 | 288 |
| dog | 644 | 356 |
| frog | 824 | 176 |
| horse | 754 | 246 |
| ship | 837 | 163 |
| truck | 866 | 134 |

## Notes

- **Training-subset accuracy is 77.69%**, only 2.4 points above test (75.32%). The model does not overfit under
  this augmentation, so its training-set errors are more numerous and less atypical than the protocol concern
  assumed. The `clean`-protocol comparison is still worth running.
- **Vehicles have the fewest errors and animals the most**, which is the usual CIFAR-10 pattern. This matters for
  PG1: the vehicle "incorrect" groups contain only 71–184 images to choose the S = 50 from.
- **Change:** `torch.load(..., weights_only=True)` for checkpoints. This removes the FutureWarning, and a
  state_dict needs nothing more.

---

# 2026-10-20 — Step 1 diagnostic: the original torchvision pipeline is also above the paper

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 1 --run-id diag_tv --backend torchvision` |
| Result | test acc **75.35%**: 2.66 points above the paper, 1.18 below our gpu backend (76.53%) |

## What it shows

- **Augmentation does not explain the gap to the paper.** The original PIL pipeline also lands far above 72.69%.
  Together with the patience analysis, the paper's lower number most likely comes from training details it does
  not report: early-stopping patience (about 1.9 points), possibly dropout, and the seed.
- **The gpu backend may be slightly weaker than torchvision** (+1.18 points). With one run each, this cannot yet be
  separated from seed noise.

## Fixed

- **ColorJitter order.** torchvision draws a new random order of the four ops for every image; `gpu_augment`
  drew one order per batch. Now drawn per sample (`data.color_jitter`). A new test compares this against
  torchvision applying the ops in the same per-sample order.

## Added

- **`steps/diag_aug_parity.py`** (`./cifar10/run.sh diag_aug_parity`). Augments the same 2,000 images 5× with
  each backend and compares per-image statistics with KS tests. The statistics are brightness, contrast,
  saturation, fill fraction, overall distortion and hue shift. A difference would point at the responsible op,
  with no training needed.
- **`--seed`** on every step. Used to measure seed-to-seed spread, so the 1.18-point gap can be judged.

## Fixed: `run.sh` exited silently

- **Symptom.** `run.sh diag_aug_parity` and `run.sh 1 --run-id seed43 --seed 43` returned immediately with no
  output.
- **Cause.** The script lookup ran `ls` on two candidate paths. One never exists (`steps/1.py`, or
  `steps/stepdiag_aug_parity_*.py`). `ls` then exits with 2, and under `set -o pipefail` the assignment kills the
  script, with stderr discarded.
- **Fix.** Check each candidate path with `[[ -f ]]`, and print an error if none matches. Tested locally with a
  stub Python: step `1` runs, `diag_aug_parity` runs, and an unknown name errors.

## Next

- [x] `./cifar10/run.sh test`: **22 passed**, including the per-sample jitter test and the rotation-vs-torchvision test
- [x] `./cifar10/run.sh diag_aug_parity`: **parity holds**, every KS < 0.05 (2,000 images × 5 augmentations per
  backend)

  | stat | torchvision | gpu | KS | p |
  |---|---|---|---|---|
  | mean | 0.3912 ± 0.1133 | 0.3943 ± 0.1132 | 0.017 | 0.13 |
  | std | 0.2556 ± 0.0609 | 0.2554 ± 0.0610 | 0.010 | 0.69 |
  | sat | 0.1102 ± 0.0740 | 0.1097 ± 0.0732 | 0.009 | 0.78 |
  | zero | 0.1145 ± 0.0758 | 0.1102 ± 0.0738 | 0.032 | 5.5e-05 |
  | dist | 0.2205 ± 0.0674 | 0.2194 ± 0.0671 | 0.012 | 0.49 |
  | hue_shift | 0.0001 ± 0.0504 | 0.0016 ± 0.0515 | 0.022 | 0.016 |

  - **`zero`** (+0.4 points of black pixels in torchvision) and **`mean`** (−0.003) are significant only because
    the samples are large.
  - Both fit how torchvision does ColorJitter: it rounds every op back to 8-bit values, so darkened near-black
    pixels round to exactly 0. Our float pipeline keeps them just above 0.
  - These are not differences in augmentation strength, so they cannot explain a 1.2-point accuracy gap. That gap
    is most likely seed noise; the seed runs will check this.
- [x] Seeds 43 and 44, gpu backend with the fixed jitter: **75.85%** and **74.76%**.

  | Run | Augmentation | Seed | Test acc |
  |---|---|---|---|
  | run01 | gpu, jitter order per batch (old) | 42 | 76.53 |
  | diag_tv | torchvision | 42 | 75.35 |
  | seed43 | gpu, jitter order per image (fixed) | 43 | 75.85 |
  | (seed 44) | gpu, jitter order per image (fixed) | 44 | 74.76 |

  - The seed alone moves the fixed gpu backend by 1.1 points, so the 1.2-point gap between backends is noise.
  - All four runs are 2.1–3.8 points above the paper.
  - **Note:** the seed 44 run reused `--run-id seed43` and overwrote its files on the server. The 75.85% result
    is kept only here. From now on, one `--run-id` per run.
- [x] **Decision: the step 1 gate is now one-sided**: test acc ≥ paper − 1.5. Deviation recorded: our dense
  Conv2Net is reproducibly about 3 points stronger than the paper's. The likely causes are unreported training
  details: early-stopping patience (about 1.9 points of it), possibly dropout, and the seed.
- [x] Retrained run01 (seed 42) with the fixed jitter: **75.32%**, step 1 gate **PASS** (one-sided). This is the
  canonical dense model for steps 2–9: `outputs/run01/conv2net_paper/step1_train/model_best.pth`.

---

# 2026-10-20 — Step 1: dense model is 3.84 points above the paper

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 1` (paper protocol, gpu backend, run01) |
| Commit | `991e3e7` |
| Result | test acc **76.53%** vs paper **72.69%** (Table 4) → gate **FAIL** (tolerance ±1.5) |
| Training | 100 epochs, all of them (the cap); one LR drop, after epoch 90; best val loss at epoch 99 |

## Diagnosis so far

**Early-stopping patience explains about half the gap.** The paper gives the LR-plateau patience (10) but not
the early-stopping patience; we assumed 15. From `history.csv`:

- The best val loss before the first LR drop was at epoch 79, with **74.67%**.
- With an early-stopping patience of 10, training would have stopped at epoch 89, before the drop at epoch 90.
  The result would have been about 74.7%, still 2 points above the paper.
- So the LR drop contributes about 1.9 points. The remaining ~2 points are present without it.

**The model is still improving at epoch 100.** It is the 100-epoch cap that ends training, not early stopping.

**Remaining candidates for the other ~2 points:**

1. **Our GPU augmentation is weaker than torchvision's**, so the model fits faster. This would be a bug, and it
   would also affect fine-tuning in step 8. A diagnostic run with `--backend torchvision` is pending.
2. **The paper's Conv2Net applied dropout before `fc`.** `cifar100/models.py` defines `Dropout(0.5)` in Conv2Net
   but never uses it, which suggests the original model did. Dropout has no parameters and ptflops does not count
   it, so step 0's exact match cannot rule it out.

## Does it matter?

Mostly no. The paper's claims are accuracy *drops* relative to each method's own dense model (Tables 1 and 5),
and the architecture is confirmed exact. It matters only if (1) is a bug.

There is one secondary risk: a better-trained model may prune differently from an under-trained one.

## Decisions

- **Stop chasing the exact 72.69% if `diag_tv` also gives about 76.5%.** Record it as a deviation and continue
  with this model.
- **Step 1 gate becomes one-sided:** test acc ≥ 72.69 − 1.5. A higher accuracy is not an implementation failure.
- **Step 8 adds a robustness check.** Rerun the two paper operating points (70% sparsity, 75.8% parameters
  removed) on a dense model near 72.7%: either the epoch-79-style checkpoint or a dropout variant.
- **Added `--es-patience`** to step 1. `gate.json` now also records `es_patience`, `backend` and `lr_drops_at`.

## Pending

- [x] `./cifar10/run.sh 1 --run-id diag_tv --backend torchvision`: 75.35% (see the entry above)
- [ ] If parity holds: switch the step 1 gate to one-sided, re-gate run01, and move to step 2
- [ ] If parity fails: fix `data.gpu_augment` before anything else

---

# 2026-10-20 — Step 0: architecture and data pipeline match the paper exactly

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 0` |
| Commit | `4409c30` |
| Result | **PASS** |

| Check | Ours | Paper (Table 4) |
|---|---|---|
| Conv2Net params | 183,434 | 183K |
| Conv2Net FLOPs (2 × MACs, ptflops) | 40,894,484 | 40.89M |
| Split sizes (paper protocol) | train 50k, val = test 10k, pg 10k | — |

The FLOP count matches to the paper's four significant figures. This confirms both that the architecture is
the same and that the paper counted FLOPs the same way, including BatchNorm (without BN the count would be
40.70M).

## Throughput

| Backend | Seconds per epoch |
|---|---|
| `gpu` (dataset on the GPU, per-sample augmentation) | **3.5** |
| `torchvision` (original PIL pipeline, 4 workers) | 8.3 |
| Old `cifar100/` pipeline, for reference | 23.3 |

At 3.5 s/epoch, the fine-tuning budget in the plan (about 7–8 GPU hours for the paper protocol) holds.

## Server housekeeping

- The server was on `cifar_100` with run 03 archive files untracked. `git pull` refused to overwrite them.
- They were moved aside, the branch was pulled, and `logs/` and `results/pruned_models/` were restored. The
  committed `README.md` (the analysis) replaced the server's empty template.
- The server's `select_model.py` toggles in `cifar100/` (conv2net vs vgg16; no algorithm changes) were stashed
  before switching to `cifar_10_Conv2Net`.

---

# 2026-10-20 — Why re-implement, and the setup decisions

## Why

The `cifar_100` work showed that the inherited pipeline does not reproduce ASF-S. The paper-vs-code comparison
(`docs/asf-s-paper-vs-implementation.md` on `cifar_100`) found several problems:

- **Eq. 12 (PGI)** multiplies three vectors from unrelated index spaces, truncated to a common length.
- **PG1** is computed with `eigh` on a non-symmetric matrix, scaled by λ^½ instead of normalised to unit length,
  and was originally built from a single sample.
- **PCA** uses 8 components where the paper uses 4.
- **Evaluation** compares a threshold-based method against ratio-based baselines, and selects checkpoints on
  the test set.

CIFAR-100 also has no paper numbers to check against. So: start again on CIFAR-10 with Conv2Net, the paper's
own setting, in steps that are each checked against a number or figure from the paper.

## Decisions

- **Location.** A new `cifar10/` directory on branch `cifar_10_Conv2Net`. It is standalone: no imports from
  `cifar100/` or `min_scripts/`.
- **Eq. 12** gets two switchable modes:
  - `literal`: the original scripts' prefix truncation, i.e. what the authors most likely ran.
  - `aligned`: dimensionally consistent; each filter is scored through the FC columns it feeds.
- **Data protocol** gets two modes:
  - `paper`: test set doubles as validation, PG1 from a training subset (what the original scripts did). Used to
    reproduce the paper's numbers.
  - `clean`: 45k/5k stratified split, test set used once. Used to report unbiased numbers.

  Running both measures the reproduction and the selection bias separately.
- **Values the paper does not give** are taken from the original scripts and marked `[assumed]` in
  `config.py`: S = 50, σ = 0.1, α = 0.5, the fine-tuning recipe, and early-stopping patience.
- **Compute.** Everything runs on the RTX 2080 server. Each step is a separate commit together with its gate
  result.
