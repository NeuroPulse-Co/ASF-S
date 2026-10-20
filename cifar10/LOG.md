# ASF-S CIFAR-10 re-implementation log

A running record of the decisions, runs, gate results and findings for `cifar10/`. The newest entry is first.
`README.md` holds the current status table. This log holds the reasoning and evidence behind it.

Each run records the host, commit, command and result. Machine: `intellisense08-EWISPro9900G`, RTX 2080,
driver 535, torch 2.5.1+cu121.

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
