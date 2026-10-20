# ASF-S CIFAR-10 re-implementation log

A running record of the decisions, runs, gate results and findings for `cifar10/`. The newest entry is first.
`README.md` holds the current status table. This log holds the reasoning and evidence behind it.

Each run records the host, commit, command and result. Machine: `intellisense08-EWISPro9900G`, RTX 2080,
driver 535, torch 2.5.1+cu121.

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
- [ ] `./cifar10/run.sh diag_aug_parity`. Expect all KS < 0.05.
- [ ] Seed spread, gpu backend: `./cifar10/run.sh 1 --run-id seed43 --seed 43` and `--run-id seed44 --seed 44`
  (about 6 min each)
- [ ] If parity holds and the seed spread covers the gap, make the step 1 gate one-sided and continue to step 2
  with the run01 model (retrained with the fixed jitter)

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
