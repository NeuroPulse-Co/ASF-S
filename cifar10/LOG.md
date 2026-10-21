# ASF-S CIFAR-10 re-implementation log

A running record of the decisions, runs, gate results and findings for `cifar10/`. The newest entry is first.
`README.md` holds the current status table. This log holds the reasoning and evidence behind it.

Each run records the host, commit, command and result. Machine: `intellisense08-EWISPro9900G`, RTX 2080,
driver 535, torch 2.5.1+cu121.

---

# 2026-10-21 — Step 8 seeds: aligned PGI's FC-only advantage is real; its full-FT advantage was noise

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 8 --scope paper --ft-seeds 42,43,44 --methods pgi_aligned,pgi_original,l1,l2 --ft-modes fc,full` |
| Runs | 24 planned, 8 done (seed 42), 16 new, about 50 min. 63 tests passed |
| Migration | `migrated 23 rows of results.csv: added ft_seed = 42` (21 paper + 2 grid rows from the stopped grid run) |
| Gate | **FAIL** on reproduction (aligned fc −10.50 vs −3.78 ± 2). PASS on ≥ L1 and ≥ random |

ΔAcc vs dense at (19, 16), mean ± sd over fine-tuning seeds 42/43/44:

| method | fc | full |
|---|---|---|
| **pgi_aligned** | **−10.50 ± 0.05** | −5.00 ± 0.42 |
| l1 | −11.99 ± 0.19 | −4.96 ± 0.62 |
| l2 | −11.91 ± 0.39 | **−4.81 ± 0.12** |
| pgi_original | −15.00 ± 0.10 | −5.61 ± 0.61 |
| random (3 draws × seed 42) | −15.07 ± 0.55 | −6.30 ± 0.48 |
| *paper Table 5, ASF-S* | | *−3.78* |

Per seed:
- **aligned:** fc −10.47 / −10.56 / −10.47; full −4.61 / −5.45 / −4.95
- **original:** fc −14.89 / −15.03 / −15.08; full −5.38 / −6.31 / −5.15

## Conclusions (these replace the single-seed reading in the entry below)

1. **FC-only fine-tuning:** aligned PGI beats L1 and L2 by 1.4–1.5 points, with seed sd ≤ 0.4. **This is a real
   effect, and the only clear win for a PGI-type criterion.**
2. **Full fine-tuning:** L2, L1 and aligned PGI fall within 0.2 points of each other, inside their seed sd. **No
   method is better.** The single-seed "aligned −4.61 is best" was noise (its seeds span −4.61 to −5.45).
3. **The authors' formula (original PGI) equals random under FC-only fine-tuning,** now over 3 seeds (−15.00 vs
   −15.07). Under full fine-tuning it is 0.6–0.8 points worse than the norm baselines.
4. **Reproduction:**
   - The FC-only claim misses by about 6.7 points (best −10.50 vs −3.78).
   - Table 5's −3.78 is about 1.0 point better than our best full-fine-tune result (−4.8 to −5.0) and 1.8 points
     better than the authors' formula.
   - Plausible sources of the remaining gap: their weaker dense model (72.69% vs our 75.32%), and a single seed
     in the paper.
5. **Limitation:** the seeds vary fine-tuning only. The pruned filters are the same for each method. Variance
   from retraining the dense model, and from the PG1 sample draw, is not included.

## Fixed

- The gate printed unrounded means (`-11.993333333333332`); seed means are now rounded to 2 decimals.

## Workflow change: server outputs move out of the repo

- **The server has no git credentials,** so results can only reach GitHub through the laptop (rsync, then
  commit). The server kept its outputs inside its working tree, so each laptop commit of results made the
  server's next `git pull` refuse to overwrite its own untracked copies. This happened three times.
- **Fix.** `config.OUTPUTS_DIR` and `run.sh`'s log folder now follow `$ASFS_OUTPUTS` (default
  `cifar10/outputs`). On the server it is set to `~/asfs_outputs`, outside the repo, so the server's tree only
  ever changes through `git pull`. The laptop rsyncs from `~/asfs_outputs/` (command in README).

## Next

- [x] Server: moved `cifar10/outputs` → `~/asfs_outputs`, set `ASFS_OUTPUTS`, restored the tracked copies, pulled `c5a75fa`
- [ ] Resume `--scope grid` (2 of 49 configs done; about 2.4 h). **Running since 10:47 on 2026-10-21, writing to `~/asfs_outputs`**
- [ ] Laptop: rsync the seed and grid results, then commit

---

# 2026-10-21 — Step 8 (paper point): FC-only recovery does not reproduce; full fine-tuning nearly does

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 8 --scope paper` (21 runs, about 65 min) |
| Point | (19, 16) kept filters: params↓ 75.8%, FLOPs↓ 82.7% (Table 5's exact configuration) |
| Dense | 75.32% test |
| Gate | **FAIL** on reproduction (aligned, fc: −10.47 vs paper −3.78 ± 2). PASS on PGI ≥ L1 and PGI ≥ random |

ΔAcc vs dense (test accuracy, percentage points):

| method | fc | fc_frozen | full |
|---|---|---|---|
| **pgi_aligned** | **−10.47** | −11.13 | **−4.61** |
| pgi_original (authors' formula) | −14.89 | −16.54 | −5.38 |
| l1 | −11.95 | −11.66 | −5.11 |
| l2 | −12.36 | −10.57 | −4.87 |
| random, mean of 3 (sd) | −15.07 (0.55) | −15.15 (0.79) | −6.30 (0.48) |
| *paper Table 5, ASF-S* | | | *−3.78 (mode not stated)* |

Before fine-tuning, every model is at 10–14%, near chance. At this sparsity, accuracy before fine-tuning says
nothing about the selection.

## Findings

1. **The paper's headline claim does not reproduce.** The claim is that fine-tuning only the output layer
   recovers baseline accuracy. The best FC-only result is −10.47 points (aligned PGI). That is a 13.9% relative
   drop, against "100% of baseline performance at sparsity of up to 70%" (§1). The paper's weaker conclusion,
   "within 10% of baseline", holds at most for aligned PGI in absolute points (−10.47) and fails for the
   authors' formula (−14.89).
2. **With full fine-tuning the numbers come close to Table 5.** Aligned PGI: −4.61 (0.83 from −3.78).
   Authors' formula: −5.38 (1.60 from −3.78, inside the ±2 tolerance).
   - Table 5 never states its fine-tuning mode. The original `min_scripts/pg_pruning_conv2.py` fine-tunes the
     **whole model** (finding #7 on cifar_100), so Table 5's −3.78 was most likely produced with full
     fine-tuning, i.e. ASF-S(a), not the FC-only (b) the paper emphasises.
   - Read that way, Table 5 roughly reproduces. The FC-only claim does not.
3. **The authors' PGI formula is no better than random under FC-only fine-tuning** (−14.89 vs random
   −15.07 ± 0.55). This matches the cifar_100 result. Under full fine-tuning it sits between L1 and random.
4. **Aligned PGI is the best criterion in every mode except fc_frozen,** but by small margins:
   - fc: +1.48 over L1, +4.6 over random
   - full: +0.26 over L2, +0.50 over L1, +1.7 over random

   With one run per method and random's spread of about 0.5–0.8 sd, the margin over L1/L2 is at noise level
   for full and about 2–3 sd for fc. **It needs repeat seeds before it can be claimed.**
5. **BatchNorm re-estimation is not what drives FC-only recovery.** fc − fc_frozen ranges from −1.79 to
   +1.65, with no consistent sign: run-to-run noise of about ±1.
6. **Full fine-tuning compresses the differences between methods** to a 2.2-point range (−4.61 to −6.82). The
   choice of filters matters much less once the remaining convs can retrain, as is common in the pruning
   literature.

## Next

- [ ] `--scope grid` (49 runs, about 2.5 h): the fc accuracy-vs-sparsity curve, and the ~72% point (26, 18) to
  compare with Table 1's Acc↓ 7.74
- [x] Repeat seeds at (19, 16) for aligned, original, L1 and L2 (fc and full), to tell whether "aligned >
  L1/L2" is real. **Added:** `--ft-seeds` and `--methods` on step 8.
  - `ft_seed` is now part of each run's identity in `results.csv`. Older files are migrated in place: existing
    rows get ft_seed = 42.
  - The pruning selection is unchanged across seeds; only fine-tuning randomness varies.
  - The tables and the gate report mean ± sd over seeds.
  - Command: `./cifar10/run.sh 8 --scope paper --ft-seeds 42,43,44 --methods pgi_aligned,pgi_original,l1,l2
    --ft-modes fc,full`. The seed-42 runs already exist, so 16 new runs, about 1 h.

---

# 2026-10-20 — Step 7 (aligned): the paper's Table 5 point is exactly (19, 16) kept filters; Table 1's is impossible

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 7 --variant aligned` (`--variant original` not run yet) |
| Gate | **PASS**: prune == mask (max \|Δlogit\| 1.1e-5); grid points within ±5 of 70% and 75.8% params↓ |

## Finding 1: Table 1's Conv2Net point is not achievable; Table 5's is, at exactly (19, 16)

- The closed-form params/MACs of a pruned Conv2Net(k1, k2) reproduce our measurements exactly: 183,434 params
  and 40,894,484 FLOPs dense, and the grid rows.
- **Table 1 (70% sparsity, 57.01% FLOPs↓) is infeasible.**
  - For every (k1, k2) with 68–72% params↓, FLOPs↓ ≥ **65.3%** (the minimum, at k1 = 32, k2 = 20).
  - Conversely, FLOPs↓ ≈ 57% allows at most **60.6%** params↓.
  - Filter sparsity gives even higher FLOPs↓.
  - So Table 1's "sparsity" is either an unstated definition or inconsistent with its FLOPs number.
- **Table 5 (75.8% params↓, 82.7% FLOPs↓) matches exactly one configuration: k1 = 19, k2 = 16** (75.84%,
  82.67%). The next-closest is off by about 0.9. So Table 5 uses parameter reduction, and the paper's best
  ASF-S Conv2Net kept 19 conv1 and 16 conv2 filters. **This is the operating point step 8 reproduces.**

## Finding 2: the τ grid is too coarse and cannot reach (19, 16)

- With the skewed aligned PGI, the 9 thresholds give only 7 distinct kept counts per layer:
  - conv1: {26, 16, 10, 5, 3, 2, 1}
  - conv2: {54, 31, 18, 12, 6, 3, 1}
- That makes 49 unique configurations. τ = 0.1 already removes 17% of parameters.
- The gate's "nearest to Table 5" point, (1, 18), keeps a single conv1 filter (FLOPs↓ 98.3%). It matches the
  parameter count only. **The nearest-point gate is a weak test and will be replaced in step 8** by top-k
  selection at the paper's exact kept counts.

## Finding 3: before fine-tuning, most pruned models are at chance

- Almost every configuration collapses to about 10% (dense: 75.32%).
- "PGI beats L1 in 71% / random in 57% of unique configs" therefore mostly compares chance-level numbers. It
  is **not** evidence.
- The informative rows (any method > 20%):

| kept (conv1, conv2) | params↓ | FLOPs↓ | PGI | L1 | random |
|---|---|---|---|---|---|
| 26, 54 | 17.2 | 30.4 | **52.08** | 42.94 | 32.50 |
| 16, 54 | 20.0 | 56.3 | 29.34 | **32.87** | 16.81 |
| 26, 31 | 52.3 | 58.3 | 21.57 | **41.98** | 17.67 |
| 26, 18 | 72.1 | 74.1 | **26.74** | 26.21 | 12.65 |

- PGI is clearly better than random where the comparison is informative, and mixed against L1. Fine-tuning
  (step 8) decides.

## Step 7 (original variant): the authors' formula is worse than L1 before fine-tuning

`./cifar10/run.sh 7 --variant original` (literal Eq. 12 prefix truncation, |mean|): gate **PASS**
(prune == mask 1.1e-5; nearest points 69.94% and 78.39% params↓). 63 unique kept configurations.

| τ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| kept conv1, original | 24 | **19** | 12 | 6 | 4 | 4 | 3 | 2 | 2 |
| kept conv2, original | 51 | 39 | 28 | 21 | 14 | 10 | 5 | 3 | 1 |
| kept conv1, aligned | 26 | 16 | 10 | 5 | 3 | 2 | 2 | 1 | 1 |
| kept conv2, aligned | 54 | 31 | 18 | 12 | 6 | 3 | 1 | 1 | 1 |

Informative rows (any method > 20% before fine-tuning):

| kept (conv1, conv2) | params↓ | FLOPs↓ | PGI (original) | L1 | random |
|---|---|---|---|---|---|
| 24, 51 | 22.3 | 38.9 | 28.05 | **42.36** | 24.80 |
| 19, 51 | 23.7 | 51.2 | 23.03 | **27.97** | 17.63 |
| 12, 51 | 25.5 | 68.4 | 14.15 | **21.58** | 13.95 |
| 24, 39 | 40.5 | 52.4 | 21.80 | **33.72** | 18.20 |
| 19, 39 | 41.6 | 62.0 | 16.32 | **27.28** | 16.43 |
| 24, 28 | 57.2 | 64.8 | 22.46 | **31.62** | 17.00 |
| 24, 21 | 67.8 | 72.7 | 10.17 | **22.55** | 14.55 |

- **The authors' PGI loses to L1 on every informative row**, by 5–14 points, and is close to random (worse
  than random at (24, 21)). This matches the cifar_100 finding that PGI ≈ random and L1 > PGI.
- **The aligned variant does better** before fine-tuning: it beats L1 at (26, 54) and ties at (26, 18).
- **Coincidence worth noting:** the original formula's τ = 0.2 keeps exactly **19** conv1 filters, the paper's
  Table 5 conv1 count. Its conv2 counts (21 at τ = 0.4, 14 at τ = 0.5) bracket the paper's 16. A different
  trained model shifts these counts, so this is consistent with the authors using the literal formula with
  τ₁ = 0.2, but it is weak evidence.
- Pre-fine-tuning accuracy is not the paper's metric; step 8 decides.

## Step 8 prepared (scope agreed: paper point + grid, paper-only available)

`asfs/finetune.py`, `steps/step8_finetune.py`, run with `--scope paper|grid|all`.

- **Paper scope (21 runs, ~1–1.5 h).** At exactly (19, 16) kept filters, chosen top-k by aligned PGI,
  original PGI, L1, L2 and random (3 draws). Each pruned model is fine-tuned in three modes:
  - **`fc`:** FC trainable, BatchNorm in train mode, so its running statistics re-adapt. This is what the
    original code does (cifar100 `fine_tune_fc` calls `model.train()` with frozen convs). It is the
    reproduction default.
  - **`fc_frozen`:** FC trainable, BN in eval mode, the pruned layers truly frozen. The strict reading of the
    paper's "freezing the pruned layers".
  - **`full`:** all layers trainable, ASF-S(a).

  The `fc` vs `fc_frozen` difference measures how much of "FC-only recovery" is really BN re-estimation.
- **Grid scope (49 runs, ~2.5 h).** The unique kept counts of the step 7 grid (aligned by default),
  fine-tuned in `fc` mode, for a Fig. 6-style accuracy-vs-sparsity plot.
- **Recipe [assumed, original scripts]:** Adam 1e-4, augmented training set, early stopping on val acc
  (patience 10, ≤ 200 epochs), best weights restored. Every run uses the same fine-tuning seed, so the
  methods are compared paired.
- **Gate (paper scope, `fc`, primary = aligned):**
  1. ΔAcc ≥ −3.78 − 2 (one-sided)
  2. PGI ≥ L1
  3. PGI ≥ mean of random

  The original PGI and all modes are reported alongside.
- **Robustness:** each run is appended to `results.csv` as it finishes, and on restart finished runs are
  skipped.

---

# 2026-10-20 — Server results imported; figure review and a step 1 correction

Pulled to this machine with
`rsync -rczm --include='*/' --include='gate.json' --include='*.png' --include='*.csv' --include='*.json' --include='*.log' --exclude='*' yutharsan@intellisense08-ewispro9900g:~/NeuroPulseCo/ASF-S/cifar10/outputs/ cifar10/outputs/`.
The server resolves over Tailscale. Checkpoints and `.npz` stay on the server.

## Correction to the step 1 diagnosis

- **The canonical run01 (75.32%) never dropped its LR** (`lr_drops_at: []`). Best epoch 98 of 100. diag_tv:
  likewise no drop, best epoch 97. seed 44 (in `outputs/seed43/`): drop after epoch 46, best epoch 91.
- **The "LR drop ≈ 1.9 points" analysis applied only to the overwritten first run01.** It does not explain the
  gap: the canonical model reaches 75.3% without any drop.
- **The models are still improving at the 100-epoch cap.** Val loss is still falling at epoch 100. Val accuracy
  stays above train accuracy throughout (train is augmented, test is not).
- **So the paper's lower 72.69% is more likely** an earlier stop in their setup, or dropout. It is not our LR
  schedule.

## Figure review

- **Fig. 3 (`fig3_afc.png`):** truck and automobile have near-identical block structure. Their own units
  (automobile, truck) are anti-aligned with a co-active block of the other 8. Dog shows a different, animal-unit
  pattern. Matches the paper's §5.2 description.
- **Fig. 4 (`fig4_pg1.png`):** PG1 picks out each class's semantic cluster.
  - Truck and automobile: + on {automobile, truck}, − on the animals.
  - Dog: + on {cat, dog, bird}.
  - Incorrect groups show the same shape plus the confusions; incorrect truck adds + on airplane and ship.
- **Fig. 5 (`fig5_contrast.png`):** the |Δg_c| scales differ about 8× between classes. Airplane, horse, deer and
  frog reach about ±0.4; dog, cat, automobile and ship stay within about ±0.1. **Without per-class
  normalisation (the paper has none), the PCA, and therefore PGI, is dominated by the few classes whose correct
  and incorrect PG1 differ most.** This is a property of the method as written. A per-class-normalised variant
  could be tested later.
- **Fig. 11 (`fig11_pgi.png`):** primary = aligned, as expected. The original variant's conv2 scores are more
  evenly spread. In every variant one or two filters set the min–max scale.

---

# 2026-10-20 — Step 6: PGI computed; the two readings of Eq. 12 pick unrelated filters

## Run

| | |
|---|---|
| Command | `./cifar10/run.sh 6` (aligned mode, Eq. 13 mean_k \|s\|) |
| Result | **PASS** (sanity gate): both layers finite, span [0, 1], all scores distinct |

| | conv1 (32 filters) | conv2 (64 filters) |
|---|---|---|
| fraction below τ = 0.1 / 0.2 / 0.3 / 0.4 / 0.5 | 0.19 / 0.50 / 0.69 / 0.84 / 0.91 | 0.16 / 0.52 / 0.72 / 0.81 / 0.91 |
| fraction below τ = 0.6 / 0.7 / 0.8 / 0.9 | 0.94 / 0.94 / 0.97 / 0.97 | 0.95 / 0.98 / 0.98 / 0.98 |
| Spearman ρ, PGI vs L1 norm | +0.224 | +0.572 |
| Spearman ρ, aligned vs original (literal, \|mean\|) | **−0.159** | **+0.033** |

## Notes

- **The interpretation of Eq. 12 decides which filters are pruned.** The dimensionally consistent reading and
  the original code's prefix truncation rank filters with essentially no correlation. Whatever the paper's
  results show was produced by the literal reading. **Step 8 fine-tunes both,** so we can see which (if
  either) reproduces Tables 1 and 5.
- **Aligned PGI is related to weight magnitude, but only partly** (ρ 0.57 for conv2, 0.22 for conv1). Whether
  the extra information helps can only be judged after pruning (steps 7–9, against L1 and random at matched
  kept counts).
- **The scores are heavily skewed.** A few filters dominate the min–max scale; half of each layer is below 0.2
  and over 90% below 0.5. Most of the 9×9 τ grid is therefore very aggressive.
- **conv2 sets the parameter count.** Each conv2 filter feeds 2,560 FC weights, and the FC holds 89% of all
  parameters. The paper's 70% operating point needs τ₂ around 0.3 (72% of conv2 pruned).

## Step 7 prepared: pruning and the τ grid (`asfs/prune.py`, `steps/step7_prune_grid.py`)

- **Physical removal.** Each kept filter is copied with its BN channel, the next conv's matching input channels,
  and, after the last conv, its 256 FC columns (channel-major). Eq. 14: keep PGI ≥ τ, and never leave a layer
  empty.
- **Correctness test: prune == mask.** The masked twin zeroes the dropped channels' *inputs to the next layer*.
  Zeroing the filter's own weights would not be equivalent, because BN + ReLU turn a zero map into the constant
  ReLU(β − γμ/σ). Tested on Conv2Net and Conv6Net at 3 keep fractions, and checked again at run time on the
  real model.
- **The grid.** 9×9 (τ₁, τ₂) for each of `--variant primary|aligned|original`. Per point it records kept
  counts, params/FLOPs reduction, filter reduction and accuracy before fine-tuning. It also records **L1 and
  random selections at the same kept counts**: an early answer to whether PGI picks better filters, with no
  fine-tuning needed.
- **Bug caught by the tests before any run.** 7 of 56 tests failed, including "keep every filter == original".
  `prune` built fresh `BatchNorm2d` modules, which start in *training* mode and so normalise with batch
  statistics instead of the copied running ones. Fix: `prune` now sets the pruned model to the source model's
  train/eval mode, and a new test checks this.
  - Step 7's accuracies would have been unaffected, because `predict()` calls `.eval()` first.
  - Step 8 is affected: fine-tuning toggles train/eval, and BN mode decides what an FC-only fine-tune
    actually updates. So this had to be fixed now.
- **Gate:** prune == mask; grid points exist within ±5 of 70% (Table 1) and 75.8% (Table 5) parameter
  reduction. Their FLOPs reductions are compared with the paper's 57.01% and 82.7%. This also tests what the
  paper means by "sparsity": parameters or filters.

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

**Early-stopping patience explains about half the gap** *(later corrected: this applied only to the first run01; the canonical run01 never dropped its LR. See "Server results imported")*. The paper gives the LR-plateau patience (10) but not
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
