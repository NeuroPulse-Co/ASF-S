# ASF-S engineering log

Running record of code-level findings, run records, and cost estimates.
Newest entry first.

> **Start with [`cifar100-status.md`](cifar100-status.md)**: it is the topic-ordered overview of
> everything below. **Paths moved on 2026-09-28:** `reference_results/` is now `original_fork/`, and
> run 01's outputs are now under `runs/run01_2026-09-16_rtx2080_S1/`. Path references below have been updated;
> the historical narrative has not.

---

# 2026-09-29 — Run 03: baselines on the fixed code

Record: [`runs/run03_2026-09-29_baselines/README.md`](../cifar100/pg_project_output/runs/run03_2026-09-29_baselines/README.md).
First run archived with `scripts/archive_run.sh` (added today).

- SVD, Sliming and SNOWS at 0.7, and HRank at 0.2–0.9: same checkpoint as run 02, fixed code, 9 rows in 112 min.
- The fork's baseline `val_acc` reproduces within −1.9 to +0.9 pts, which confirms that `rebuild_fc` barely
  matters once the FC is retrained.
- Pareto comparison with PG's 81 rows: **PG is better only at light pruning** (it beats HRank 0.2 and 0.3).
  At ≥ 50%, Sliming, SVD and HRank 0.5–0.8 dominate points on PG's front. Sliming at 10/19 (70.5% params,
  87.8% FLOPs) scores 37.02 against PG 10/20 (69.0%, 87.2%) at 32.00.

---

# 2026-09-29 — Run 02 results: fixes work, ASF-S does not reproduce on CIFAR-100

Full record: [`runs/run02_2026-09-28_rtx2080_S50_fixes/README.md`](../cifar100/pg_project_output/runs/run02_2026-09-28_rtx2080_S50_fixes/README.md).

- **Steps 2, 4 and 6 completed on the server** (step 6: 81 combinations in 1024 min, not the projected ~50 h).
  The dump is now 8.8 GB at 474 KB per file. That confirms finding #13's mechanism: ≈460 KB of
  activations plus the 12 KB input.
- **PG1 is healthy:** 0 constant vectors (was 26), PCA PC1 0.388 (was 0.050), and W well connected
  at σ = 0.1.
- **`rebuild_fc` checked on the real checkpoint:** 52.36% with 0 filters pruned, the same as unpruned.
  Randomly dropping (1, 2) filters gives 48.03%.
- **Best FC-only results:** 51.12% at 11% sparsity, **40.46% at 68.7% (−12.1 pts)**. That is within about 0–2
  pts of the fork's results at the same sparsity, so the fixes did not change final accuracy.
- **PG vs random vs L1 at identical kept counts** (no fine-tuning, 8 rows, 10 random draws each): PG averages
  only +1.8 pts over random, and 30 of 80 draws matched or beat it. L1-norm beats PG in 7 of 8 rows. After
  fine-tuning, the fork's Sliming, SVD and SNOWS at 10/19 kept (36.2, 35.9, 33.7) all beat PG at 10/20 (32.0).
- **Noise floor:** combinations with identical kept counts differ by 0.29 on average and 0.73 at most in `val_acc`.

Interpretation: PGI is not selecting informative filters. The most likely cause is Eq. 12 (#4/#4b), which
the user decided to leave unchanged for this run. Next: re-run the baselines on the fixed code, and raise
Eq. 12 with the authors.

---

# 2026-09-28 — Minimal fixes for run 02: `rebuild_fc`, param counts, S = 50

Findings #1–#3 fixed in code. Everything else is intentionally left alone, so run 02 isolates
their effect. The full list and rationale are in
[`cifar100-status.md`](cifar100-status.md#code-changes-for-run-02).

- **#1:** `rebuild_fc(model, fc_attr, keep_idx)` now copies the kept channels' FC columns; every
  caller passes the last conv's kept `idx`. It raises on a shape mismatch instead of silently
  mis-wiring.
- **#2:** `count_params()` replaces `ptflops`' param count (which skips frozen params) for both
  the base and pruned models in PG, HRank and `benchmark_pruning_common.py`.
- **#3:** `pg_ext_cifar100.py` stacks up to `PG_SAMPLES = 50` samples per (class, split), with a seeded
  shuffle. `pg1_summary.csv` gains `n_samples`, `pg1_distinct`, `w_offdiag_mean` and `w_frac_gt_0.01`.
- **Also:** PG1 sign alignment before Eq. 9 (an implementation choice, stated as such), and
  `inputs[i].clone()` in activation extraction (dump size).

**Verified on CPU with synthetic inputs:**
- With random channel subsets pruned, the pruned Conv2Net, Conv6Net and VGG16 match the unpruned
  model with those channels silenced (max |Δ| ≈ 1e-7). The pre-fix code fails the same check.
- `count_params` is correct with layers frozen.
- `pg_ext` gives 448–512 distinct PG1 values at S = 50.

**Not yet verified on real data.** That needs steps 2 and 4 on the server.

Also noted in the 2026-09-28 re-check, but not in this log before: the VGG16 baseline in
`vgg16_metrics.csv` is 58.17% best / 55.99% final. The "57.14%" under finding #8 below does not
match the file.

---

# 2026-09-18 — Finding #3 confirmed empirically: PG1 has collapsed

Steps 2 and 4 completed on the department server after the storage problem was
addressed server-side (step 2 took 13 min against the 45-70 min estimate; the working
configuration was not recorded and should be, since dump size now gates VGG16).

Step 4 processed all 200 (class, split) pairs in **2.7 seconds** — itself a symptom,
since it reads one file per pair.

The resulting `pg_data/` was copied to a laptop and inspected. **Finding #3 moves from
predicted to confirmed**, and the reality is worse than the theory suggested.

## PG1 vectors are two-valued

Each vector has 100 entries (Conv2Net's `fc` is the 100-class output layer) and takes
**1-2 distinct values**: mean 1.96 across the `correct` set, 1.74 across `incorrect`.

```
class 1 correct    -1.003850e-02  x99        class 1 incorrect  -1.003891e-02  x99
                    9.949943e-01  x1                             9.949939e-01  x1
```

A 99/1 split — one unit separated from the other 99. This is precisely the two-block
affinity matrix predicted for S=1: what is being called a principal gradient is the
sign partition of a single image.

Two aggravating details the theory did not predict:

- **26 of 100 `incorrect` vectors are constant** (a single value across all 100 units).
  Those contribute nothing whatsoever to the Eq. 9 contrast.
- **`correct` and `incorrect` are near-identical for some classes.** In class 1 above
  they agree to six significant figures, so `Δg = g_corr − g_incorr ≈ 4e-7` — numerical
  noise, for the quantity the whole method is built on.

## The contrast matrix has no dominant direction

Stacking to ΔG (100×100, Eq. 10) and applying Eq. 11's PCA:

| | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
|---|---|---|---|---|---|---|---|---|
| explained variance | 0.050 | 0.046 | 0.040 | 0.039 | 0.037 | 0.034 | 0.033 | 0.032 |

Nearly flat — a factor of 1.6 between first and eighth, against 0.01 for a perfectly
uniform spectrum. **PCA(8) captures 30.9% of variance; 75 components are needed for
99%.** ΔG has rank 99 of 100.

Eq. 11 claims to extract "the dominant, class-shared axes". There is no dominant axis
present to extract, so the PCA components fed into Eq. 12 are close to arbitrary.

For context the paper's §C.3 reports 4 components accounting for ~90% of variation.
That is **not** a like-for-like contradiction — it describes CIFAR-10 with a 10-unit
output layer, where 4 of 10 components reaching 90% is unremarkable — but the CIFAR-100
extension is nowhere near that regime, and the discrepancy should be explained rather
than inherited.

## What this does and does not establish

**Established:** the PG1 vectors are degenerate, and the PCA step finds no concentrated
direction from which to build PGI.

**Not established:** that ASF-S does not work. A near-arbitrary importance score can
still outperform random pruning by accident, and the committed reference results do
show 50.03% recovery at light pruning. Whether the method survives a correctly computed
PG1 is exactly what the S=50 re-run answers.

**A null model was tried and does not support a stronger claim.** Random two-valued
rows of the same shape produce *more* spectral concentration than the real ΔG
(PC1 ≈ 0.50 vs 0.05), because random row means create a shared offset direction. So
"indistinguishable from noise" is not a claim the data supports; the narrower claim
above is.

## Reproducing this

From `pg_data/` alone, no GPU required:

```python
import numpy as np, glob, os
base = 'cifar100/pg_project_output/runs/run01_2026-09-16_rtx2080_S1/pg_data/conv2net'
def load(split):
    fs = sorted(glob.glob(f'{base}/{split}/pg1_data/*_fc_pg1.npy'),
                key=lambda p: int(os.path.basename(p).split('_')[0]))
    return np.stack([np.load(f) for f in fs])

C, I = load('correct'), load('incorrect')
print([len(np.unique(np.round(r, 6))) for r in C][:10])   # distinct values per class
print(sum(len(np.unique(np.round(r, 6))) == 1 for r in I))  # constant vectors

dG = C - I
X  = dG - dG.mean(0)
s  = np.linalg.svd(X, compute_uv=False)
ev = s**2 / (s**2).sum()
print(np.round(ev[:8], 4), ev[:8].sum())
```

---

# 2026-09-16 — Step 1 reproduces; step 2 blocked by disk

Environment: `intellisense08-EWISPro9900G`, NVIDIA RTX 2080 (8 GB), driver 535.309.01
(CUDA 12.2), torch 2.5.1+cu121, Python 3.10. Run via
`./scripts/run_pipeline.sh --steps 1`, which selected conv2net and trained for 64 min.

## Conv2Net / CIFAR-100 training vs the reference run

| | New (RTX 2080) | `original_fork/` |
|---|---|---|
| Epochs run | 165 | 160 |
| Best `val_acc` | **0.5252** @ epoch 152 | **0.5193** @ epoch 156 |
| Final train / val acc | 0.4885 / 0.5237 | 0.4863 / 0.5171 |
| NaNs in `train_loss` | 0 | 0 |
| Early stop (cap 200) | yes | yes |

**Verdict: reproduces.** +0.59 accuracy points is ordinary run-to-run variance from
nondeterministic cuDNN kernel selection on different hardware. Early stopping fired at
a similar epoch, so the run converged the same way rather than stalling or being
truncated. `val_acc` sitting above `train_acc` is expected — dropout and augmentation
are active during training and off at evaluation — and the gap matches the reference.
No NaNs, so AMP behaves on Turing.

Minor oddity, not a defect: epoch 1 reached 19.6% val accuracy, unusually strong for
CIFAR-100. The curve is smooth from there to 52%, so it reads as a lucky
initialisation.

## Timing recalibrated on real hardware

64 min / 165 epochs = **23.3 s per epoch**, against ~19 s on the Kaggle T4. The RTX
2080 is *slower* here despite being the stronger card, which indicates the workload is
**CPU-bound in the DataLoader**, not GPU-bound. Relevant because the scripts hardcode
`num_workers=4`.

Scaling the measured Kaggle step-6 cost by that ratio:

| | Kaggle T4 | RTX 2080 (projected) |
|---|---|---|
| Per combination | ~30 min | **~37 min** |
| 81 combinations | ~40-45 h | **~50 h** |

Feasible in one tmux session with no session limit, but it is two days of GPU time.
Two consequences:

1. Decide up front whether all 81 combinations are wanted, or a coarser grid first.
2. This strengthens the case for fixing finding #1 (`rebuild_fc`) **before** the run —
   50 hours is a lot to spend producing a table whose `pre_acc` column cannot mean
   what it claims.

Estimate accuracy so far: step 1 was predicted at ~25 min and took 64. Projections in
this log should be treated as order-of-magnitude until measured on this box.

## Step 2 — failed, `/home` full

```
RuntimeError: [enforce fail at inline_container.cc:778] .
  PytorchStreamWriter failed writing file data/0: file write failed
RuntimeError: [enforce fail at inline_container.cc:603] . unexpected pos 704 vs 598
```

Died at class 66 of 100 during `torch.save`. The second error is the truncated write's
end-of-file record — i.e. a corrupt `.pt` was left behind. Step 2 has no resume, so a
re-run restarts at class 0 and will not necessarily overwrite it; the partial dump must
be deleted rather than reused, or a later step hits it as an unpickling error.

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdb1       1.6T  1.5T     0 100% /home
```

`/home` is a **shared 1.6 TB volume at 100%, 0 bytes available** — so this blocked every
user on the box, not just this run. No quota is configured; of the 1.5 TB in use only
26 GB was ours. Deleted to release it.

### The activation dump is far larger than estimated

| | Predicted | Actual |
|---|---|---|
| Per file | ~460 KB | **~2 MB** |
| Full conv2net dump | 9.4 GB | **~39 GB** (26 GB at class 66/100) |

The ~4× gap against the per-layer arithmetic (conv1 32·32·32 + conv2 64·32·32 +
pool 64·16·16 + fc 100 = 114,788 floats ≈ 460 KB) is **unexplained and worth
investigating**. A plausible culprit is `torch.save` serialising a whole shared storage
rather than the sliced view in `{n: a[i].cpu() for n, a in acts.items()}`
([`activation_ext_cifar100.py:66`](../cifar100/activation_ext_cifar100.py#L66)) — each
`a` is a full `(batch, C, H, W)` tensor and `a[i]` is a view into it. If that is the
cause it is a real and cheaply fixed 4× waste.

Consequence: the ~25 GB VGG16 figure in the 2026-09-14 audit is also unreliable and
likely nearer 100 GB. Both need measuring, not recalculating.

### Almost all of the dump is unused on the path being run

With step 3 skipped, the only consumer of these files is `pg_ext_cifar100.py`, which
reads **one layer** (`PG_FC_LAYER["conv2net"] = "fc"`, 100 floats) and **one file per
(class, split)** — the latter being finding #3, the `break` on first match.

So `conv1`, `conv2`, `pool`, and 199 of every 200 files exist solely to feed step 3's
affinity matrices. Saving only `fc` would take the dump from ~39 GB to roughly
**250 MB**. Worth adding as an opt-in (`PG_SAVE_LAYERS`) defaulting to current
behaviour, so step 3 remains possible later on a machine with room.

### Preflight defect

`run_pipeline.sh` *warns* below 15 GB free but does not abort, and 15 GB was far too low
regardless. It should abort, and require ~45 GB for conv2net — a measured figure, not a
calculated one.

**Status: blocked pending a server-side storage decision** — another filesystem, or
shrinking what step 2 writes.

---

# 2026-09-15 — Kaggle abandoned, moving to the department GPU server

## Why

The Kaggle run on 2026-09-13 **timed out at the 12h limit**, having completed steps
1-5 and only 15 of 81 combinations in step 6 — it died partway through #16 (full
record below). A timed-out Kaggle commit does not save its output, so the trained
checkpoint and all partial results were lost.

The root cause is not Kaggle-specific: step 6 needs ~40-45 GPU-hours for the
Conv2Net threshold grid, which exceeds both the 12h session limit and the 30h/week
quota. No amount of tuning makes that fit.

## New target

Department GPU server `intellisense08-EWISPro9900G` (Ubuntu 22.04), sole user for the
duration of the run. This removes the two constraints that shaped the Kaggle plan:

- **No session limit.** The 40-45h grid can run to completion in one `tmux` session.
- **Partial results survive a crash.** Results append to CSV per combination and the
  filesystem persists, so an interrupted run keeps everything finished up to that
  point — unlike the Kaggle commit, which discarded all of it.

Consequently the two-notebook split and the CSV resume logic proposed in the Kaggle
run record below are **no longer needed**. They were workarounds for the session
limit.

## Repo change made for this: `results/` → `reference_results/`

The scripts append with `header=not file_exists`
([`pg_pruning_cifar100.py:178`](../cifar100/pg_pruning_cifar100.py#L178)), and a fresh
clone contains 16 committed CSVs in `results/`. New rows were therefore being appended
underneath the old ones with no header or marker between runs — this already happened
on Kaggle, where the 16 completed combinations landed under the 9 pre-existing rows of
`pg_conv2net_cifar100.csv`.

All 17 previously committed result files were moved to
`cifar100/pg_project_output/reference_results/` (since renamed to `original_fork/`; named to avoid colliding with
`base_*` in the code, which means the *unpruned* model, and with the README's
"baselines", which means SVD/Sliming/SNOWS).
`config.py` recreates an empty `results/` on import, so new runs write cleanly with a
proper header. Verified: new `results/*.csv` are **not** gitignored and so remain
committable, while `results/pruned_models/` is still ignored by `.gitignore:15`.

## Operational notes for the server

- Run under `tmux` — a dropped SSH session otherwise kills a 40h job.
- Use a project-local venv, not the shared system python.
- Budget ~11 GB: the Conv2Net activation dump alone is ~9.4 GB.
- Pull results back with `rsync`, excluding `activations/` (intermediate, never needed
  off-box) and optionally `results/pruned_models/` (~250 MB).
- Copy `trained_models/conv2net_best.pth` off as soon as step 1 finishes. It is ~25
  min of GPU time and every later step depends on it.
- The Kaggle notebook does not transfer directly — it is built around
  `/kaggle/working` and a `/kaggle/temp` symlink.

## Server runner added

[`scripts/run_pipeline.sh`](../scripts/run_pipeline.sh) replaces the notebook. It runs
the repo's scripts unchanged; only model selection is automated, via
[`scripts/select_model.py`](../scripts/select_model.py).

```bash
tmux new -s asfs
./scripts/run_pipeline.sh --setup                 # venv, once
./scripts/run_pipeline.sh --steps 1               # train, ~25 min; copy the ckpt off
./scripts/run_pipeline.sh --steps 2,3,4,5,6       # the rest
```

Preflight checks tmux, CUDA, free disk and whether anyone else is on the GPU; each
step is timed and tee'd to `logs/<run-id>/`; step 1 is skipped if a checkpoint already
exists; and it warns if a `results/*.csv` for the chosen model already exists, since
the scripts would append to it with no header.

### Torch wheel vs driver — an install-time trap

`cifar100/requirements.txt` pins only `torch>=2.0.0`, so `pip install -r
requirements.txt` fetches whatever wheel is newest. On the department server
(RTX 2080, driver 535.309.01 = CUDA 12.2) that was `torch 2.14.0+cu130`, which
imports fine but fails at first CUDA call with:

```
The NVIDIA driver on your system is too old (found version 12020)
```

This is silent until runtime and would otherwise be discovered part-way into a long
job — or worse, not at all if something falls back to CPU.

`--setup` now reads the CUDA version from `nvidia-smi` (the newest runtime the driver
supports), maps it to a wheel index, and installs torch from there *before* the
requirements file, whose `>=` constraints then leave it alone:

| Driver reports | Index | Pinned |
|---|---|---|
| ≥ 12.8 | cu128 | unpinned |
| ≥ 12.6 | cu126 | torch 2.7.1 / tv 0.22.1 |
| ≥ 12.4 | cu124 | torch 2.6.0 / tv 0.21.0 |
| ≥ 12.1 | cu121 | torch 2.5.1 / tv 0.20.1 |
| ≥ 11.8 | cu118 | torch 2.4.1 / tv 0.19.1 |
| none | cpu | unpinned, with a warning |

Override with `--cuda cu121` etc. Setup verifies `torch.cuda.is_available()` and fails
loudly rather than leaving a venv that only breaks later. Mapping verified against
stubbed driver versions 11.6 through 13.0.

**Working combination on the department server: torch 2.5.1+cu121, Python 3.10.**

`select_model.py` is idempotent and verified to round-trip byte-identically across all
three models. Useful side effect: run with `--check` it reports the current selection,
and on the pristine repo it correctly returns `mixed/unknown` — `hrank_cifar100.py`
was left on `conv2net` while the other four files were on `vgg16`.

## Still open

The audit findings below are unaffected by the change of hardware. In particular
finding #1 (`rebuild_fc`) should be fixed **before** committing 40 GPU-hours to a
grid, since it determines what the resulting numbers mean.

---

# 2026-09-14 — Code audit against the paper, and the first Kaggle run

Context: attempting to reproduce `cifar100/` on Kaggle raised questions about the
pipeline, so the code was read against *Artificial Structure Function Search:
Preserving Artificial Functional Connectivity for Structured Pruning* (NeurIPS 2026
submission). Eleven issues below, ordered by how much they affect reported results.

Every claim marked **Verified** was checked by arithmetic against CSVs already
committed in `cifar100/pg_project_output/results/`, so it is reproducible without
a GPU and is not an artefact of the Kaggle environment.

## Severity summary

| # | Finding | Affects | Status |
|---|---|---|---|
| 1 | `rebuild_fc` writes weights to the wrong columns | every `pre_acc`; the "no retraining" claim | **Verified** |
| 2 | Pruned param counts exclude conv layers | every `param_red_pct` / sparsity figure | **Verified** |
| 3 | PG1 is built from a single sample | every PGI score and pruning result | **Verified + empirically confirmed 2026-09-18** |
| 4 | Eq. 12 projection truncated to the shortest vector | every PGI score | **Verified** |
| 5 | PG threshold is not a sparsity ratio | PG-vs-baseline comparisons | **Verified** |
| 6 | Legacy importance uses `\|mean(s)\|`, Eq. 13 says `mean(\|s\|)` | CIFAR-10 results only | **Verified** |
| 7 | `min_scripts` fine-tunes the whole model | which figure the CIFAR-10 numbers belong to | **Verified** |
| 8 | VGG16 cosine LR schedule is never stepped | VGG16 baseline accuracy | **Verified** |
| 9 | Two SVD CSVs have a corrupted header line | reading those results | **Verified** |
| 10 | Activation extraction is O(classes × dataset) | runtime only | **Verified** |
| 11 | Root `README.md` is stale | onboarding only | **Verified** |

---

## 1. `rebuild_fc` copies FC weights into the wrong columns

**Severity: highest.** This one bears directly on the paper's headline claim.

### Evidence

Every row of the committed `pg_conv2net_cifar100.csv` has `pre_acc` at or below
chance (1.0% for CIFAR-100), no matter how little was pruned:

```
rows=9   pre_acc min/max = 0.66 / 1.83   val_acc min/max = 8.40 / 50.03
kept_conv1=31/32 (3.1% pruned), kept_conv2=58/64 (9.4% pruned) -> pre_acc = 1.83
```

Baseline Conv2Net is **51.93%** (`conv2net_metrics.csv`, epoch 156). Removing 3% of
conv1 and 9% of conv2 filters cannot cost 50 accuracy points. The same pattern holds
across every method — `svd` 0.79, `sliming` 1.34, `pg_vgg16` 1.00 — because they all
route through the same helper.

### Mechanism

[`pruning_utils.py:83-87`](../cifar100/pruning_utils.py#L83-L87):

```python
new = nn.Linear(size["n"], old.out_features).to(device)   # randomly initialised
m   = min(old.in_features, size["n"])
new.weight.data[:, :m] = old.weight.data[:, :m]           # first m columns
```

The flattened conv feature vector is channel-major: for Conv2Net, feature index
`= c*256 + r*16 + col`. Taking the **first** `m` columns therefore copies the weights
of old channels `0 .. k-1`. But the kept channels are wherever PGI happened to place
them. Unless the kept set is exactly `{0..k-1}`, every copied weight lands on a
feature it does not belong to, and any column past the copy keeps its random
initialisation.

### Impact

- **`pre_acc` measures nothing.** It is the accuracy of a network whose classifier has
  been arbitrarily rewired, not of the pruned network. Any "post-prune, no recovery"
  number for the CNNs — the dashed curves in Figure 7 — cannot be supported by this
  code path.
- **It changes what the recovery experiment means.** Because the FC starts from
  effectively random weights, "recover baseline accuracy by fine-tuning only the
  output layer" is measuring a linear probe retrained on frozen pruned conv features.
  That is a real and defensible experiment, but it is not evidence that AFC was
  preserved — which is the claim in §4 and the abstract.
- **Comparisons between methods are probably still fair.** Every method (PG, HRank,
  SVD, Sliming, SNOWS) goes through the identical broken path, so the *relative*
  post-fine-tune `val_acc` ordering may well survive. The invalid parts are the
  `pre_acc` column and the interpretation of the recovery result.

### Fix

Index the columns by kept channel instead of taking a prefix. Given `keep_idx` (the
kept channel indices of the last conv) and spatial size `H*W`:

```python
cols = np.concatenate([np.arange(c*H*W, (c+1)*H*W) for c in keep_idx])
new.weight.data = old.weight.data[:, cols].clone()
```

This makes the pruned network function-preserving up to the removed filters, at which
point "no retraining of the pruned layers is needed" becomes a testable claim rather
than an assumed one. **Re-running with this fix is the single highest-value
experiment available.**

---

## 2. Pruned parameter counts exclude every conv layer

### Evidence

From the Kaggle run log (combo 15/81, `kept=[23, 11]`):

```
Conv2Net(
  281.7 k, 100.000% Params, 3.36 MMac, 98.649% MACs,
  (conv1): Conv2d(0, 0.000% Params, 659.46 KMac, 19.333% MACs, 3, 23, ...)
  (conv2): Conv2d(0, 0.000% Params, 2.34 MMac, 68.686% MACs, 23, 11, ...)
  (fc):    Linear(281.7 k, 100.000% Params, 281.7 KMac, 8.258% MACs, in=2816, out=100)
)
```

`2816 * 100 + 100 = 281,700` exactly — the reported total is the FC layer alone.
The convs contribute 88% of MACs but 0% of params.

Confirmed in already-committed data, independent of the Kaggle run. For
`pg_conv2net_cifar100.csv` row `kept=[31,58]`:

```
58 * 256 * 100 + 100          = 1,484,900   == the CSV's `params`
Conv2Net total (unpruned)     = 1,658,084
1 - 1484900/1658084           = 10.4448%    == the CSV's `total_param_red_pct`
```

Four-decimal match on both.

### Mechanism

`ptflops` appears to count only parameters with `requires_grad=True`.
[`pg_pruning_cifar100.py:218-219`](../cifar100/pg_pruning_cifar100.py#L218-L219)
freezes the convs and BNs, and `fine_tune_fc` freezes everything except the FC
attrs — all *before* line 236 measures the pruned model. But `base_params`
(line 150) was measured on the freshly loaded model with everything trainable.
The two measurements are not comparable.

Same ordering in [`hrank_cifar100.py:128-134`](../cifar100/hrank_cifar100.py#L128-L134)
and [`benchmark_pruning_common.py:207-214`](../cifar100/benchmark_pruning_common.py#L207-L214),
so it affects the baselines too.

Confirm in five lines:

```python
from models import VGG16CIFAR
from ptflops import get_model_complexity_info
m = VGG16CIFAR(100)
print(sum(p.numel() for p in m.parameters()))                  # true total
for p in m.conv1_1.parameters(): p.requires_grad = False
print(get_model_complexity_info(m, (3,32,32), as_strings=False, verbose=False)[1])
```

If the second number drops by 1,792, the mechanism is confirmed.

### Impact

Depends entirely on how much of the model lives in the FC layer:

- **Conv2Net: negligible.** Convs + BNs are 19,584 of 1,658,084 params (1.2%), so
  every conv2net sparsity figure is off by at most ~1.2%. The committed conv2net
  results stand on this axis.
- **VGG16: severe.** Convs are 14,714,688 of 15,037,092 params (97.9%). At the one
  committed row (thr=0.7, 99.53% reported) almost all convs are gone anyway, so the
  true figure is ~99.46% — close. But at *moderate* sparsity, where the convs still
  hold most of the parameters, the reported reduction will read far higher than the
  truth. **No VGG16 row at moderate sparsity has been produced yet**, so nothing is
  currently mis-reported — but every future one would be.
- **FLOPs are unaffected.** MACs are counted correctly (conv1 shows 19.3% MACs).

### Fix

Measure both the baseline and the pruned model in the same state — either unfreeze
before calling `get_model_complexity_info`, or count params directly with
`sum(p.numel() for p in m.parameters())`.

---

## 3. PG1 is built from one sample per (class, split)

### Evidence

Both extractors break out of the file loop on the first match:

- [`min_scripts/pg_ext_2.py:83`](../min_scripts/pg_ext_2.py#L83) — `break  # only 1 sample needed`,
  with `max_samples = 50` declared at line 29 and **never used** in that function
- [`cifar100/pg_ext_cifar100.py:108`](../cifar100/pg_ext_cifar100.py#L108) — same

So `acts` is `(1, units)` and `cosine_affinity_gpu(acts.T)` receives `(units, 1)`:
Eq. 1's `X ∈ ℝ^{N×S}` has **S = 1**.

### Consequence

With S=1, `F.normalize` maps each unit to ±1, so Eq. 2 gives `A_ij ∈ {−1, +1}`.
At σ=0.1, Eq. 3 gives `W_ij = 1` for same-sign pairs and `exp(−200) ≈ 0` otherwise.
W collapses to a two-block matrix, and PG1 becomes the **sign pattern of one image's
activations** rather than the co-fluctuation structure §3 defines ("each unit produces
an activation vector across samples, which can be treated as the analogue of a neural
time series").

On post-ReLU layers it degenerates further: all live units have `cos = +1` exactly and
dead units `cos = 0`, so PG1 reduces to an alive/dead indicator. This applies to the
maxpool panels in Figures 15-18.

Note the AFC maps are **not** affected: `cosine_sim_ext_2.py:113-125` does stack 50
samples, so Figures 3, 13 and 14 follow Eq. 1-3 as written. The split is Figure 3
(correct) vs Figure 4 and everything downstream of it (S=1).

### Cheapest check

```python
np.unique(np.round(np.load('cifar100/pg_project_output/runs/run01_2026-09-16_rtx2080_S1/pg_data/conv2net/correct/pg1_data/0_fc_pg1.npy'), 6))
```

A 100-element PG1 vector returning two or three distinct values confirms the collapse.

> **Run on 2026-09-18: confirmed.** Every vector is 1-2 valued, 26/100 `incorrect`
> vectors are constant, and PCA(8) over the contrast matrix captures only 30.9% of
> variance. See the 2026-09-18 entry.

---

## 4. Eq. 12's projection is truncated to the shortest vector

`⟨w_i^(ℓ), u_k ⊙ w̄^(f)⟩` requires `dim(w_i^(ℓ)) = dim(u_k) = dim(w̄^(f))`, but the
three differ. The code resolves this by prefix-truncation to the shortest
([`pg_pruning_cifar100.py:104`](../cifar100/pg_pruning_cifar100.py#L104),
`min_scripts/pg_pruning_conv2.py:99`):

```python
sz   = min(conv.shape[1], avg_fc.shape[0], comp.shape[0])
proj = conv[:, :sz] @ (comp[:sz] * avg_fc[:sz])
```

For **Conv2Net on CIFAR-10**: `u_k ∈ ℝ¹⁰` (PG1 on the 10-unit output layer),
`w̄ ∈ ℝ¹⁶³⁸⁴`, conv1 filters `∈ ℝ²⁷` → `sz = 10`. PGI uses only the **first 10 weights**
of each filter, dotted against the 10 FC-mean entries for whatever flattened spatial
positions happen to come first.

For **VGG16 on CIFAR-100** it varies by layer: 27 for `conv1_1`, 512 for `conv5_3`.

The paper does not specify how this mismatch should be resolved. A prefix is an
arbitrary choice and should either be justified or replaced with something
principled.

---

## 5. PG prunes by threshold, baselines prune by ratio

`pg_pruning_cifar100.py` keeps filters with `importance > threshold` on a min-max
normalised score. `benchmark_pruning_common.py` prunes to an exact ratio. Table 1
compares at matched 70% sparsity, but a PG threshold of 0.7 is not 70% sparsity:

```
pg_vgg16_cifar100.csv, thr=0.7 -> 99.53% params removed, val_acc 1.71%
```

Chance is 1%. Any comparison table must select the PG threshold whose **achieved**
sparsity matches the baseline ratio, not reuse the knob value.

---

## 6. Legacy importance formula differs from Eq. 13

Eq. 13 is `N( (1/K) Σ_k |s_i,k| )` — mean of absolute values.

- `min_scripts/pg_pruning_conv2.py:105`: `np.abs(scores.mean(axis=0))` — **absolute of the mean**
- `cifar100/pg_pruning_cifar100.py:108`: `torch.stack(scores).abs().mean(dim=0)` — matches Eq. 13

The CIFAR-10/MNIST results in the paper were produced with the first form, which
allows cancellation between components before the absolute value is taken.

---

## 7. `min_scripts` fine-tunes the whole model

[`min_scripts/pg_pruning_conv2.py:192`](../min_scripts/pg_pruning_conv2.py#L192) uses
`optim.Adam(model.parameters(), lr=1e-4)` with `p.requires_grad = True` for every
parameter. That is ASF-S(a) in Figure 6 — full fine-tuning.

The FC-only variant ASF-S(b), which is the paper's headline claim, exists only in
`cifar100/pg_pruning_cifar100.py`. Worth confirming which script produced which
figure before the camera-ready.

Related: `min_scripts` cannot currently run — it imports
`from models import Conv2Net, Conv6Net, LeNet300_100` and that `models.py` is not in
the repo, nor is there a training script for those checkpoints.

---

## 8. VGG16's cosine schedule is never stepped

[`train_cifar100.py:137-139`](../cifar100/train_cifar100.py#L137-L139) builds
`sch_cos` then passes `scheduler=None`, with a comment saying it will be stepped
manually — it is not. VGG16 trained at a flat lr=0.05 for 108 epochs and reached
57.14% (`vgg16_metrics.csv`), well short of a properly scheduled VGG16 on CIFAR-100.

---

## 9. Two SVD result CSVs have a corrupted header line

A missing newline glued the first data row's `method` value onto the end of the
header:

| File | header fields | data fields | header ends with |
|---|---|---|---|
| `svd_conv2net_cifar100.csv` | 18 | 16 | `pruned_conv2_pctsvd,` |
| `svd_conv6net_cifar100.csv` | 26 | 24 | `pruned_conv6_pctsvd,` |

`svd_vgg16` (39/39) and the sliming/snows files are fine. Recoverable by hand: strip
the trailing `svd,` from the header and prepend `svd,` to the data row.

---

## 10. Activation extraction rescans the dataset once per class

[`activation_ext_cifar100.py:88`](../cifar100/activation_ext_cifar100.py#L88):

```python
indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == class_id]
```

inside the `for class_id in range(NUM_CLASSES)` loop. This decodes and transforms the
entire 100k-sample subset 100 times — 10M image decodes. `datasets.CIFAR100` exposes
`.targets`, which makes the index map one line. Runtime only, no effect on results.

---

## 11. Root `README.md` is stale

It describes `CNN/cifar100_original_pipeline/main_cifar100_original.py` and a `lenet`
model, none of which exist, and states that VGG is not part of the pipeline — the
opposite of the current code. `cifar100/README.md` is the accurate one.

---

# Run record — Kaggle, Conv2Net / CIFAR-100, 2026-09-13

Notebook: [`notebooks/kaggle_conv2net.ipynb`](../notebooks/kaggle_conv2net.ipynb)
Environment: Kaggle, 2× Tesla T4, torch 2.10.0+cu128, Python 3.12.13

**Outcome: timed out at the 12h limit, at combination 16 of 81 in step 6.**

Steps 1-5 completed. `torch.cuda.amp` still works in torch 2.10 (no import error),
so no AMP patch was needed.

## Revised cost model

The pre-run estimate of ~7h for step 6 was wrong by roughly 5×.

| Quantity | Estimated | Actual |
|---|---|---|
| FC fine-tune epoch | ~8 s | **~19 s** |
| Epochs per combination | ~40 | ~60-70 (early stop, patience 10, max 200) |
| Per combination | ~5 min | **~30 min** |
| Step 6, 81 combinations | ~7 h | **~40-45 h** |

The 81 combinations come from `itertools.product(THRESHOLDS, repeat=2)` for Conv2Net
([`pg_pruning_cifar100.py:184`](../cifar100/pg_pruning_cifar100.py#L184)). At 40-45
GPU-hours this exceeds both the 12h session limit and the 30h/week Kaggle quota, and
cannot complete in one run.

## Structural problems this exposed

1. **No resume.** Results append to CSV as they go, but there is no skip-completed
   logic, so a re-run restarts at combination 1 and repeats finished work.
2. **All-or-nothing session.** Steps 1-4 (~4h, producing the checkpoint and `pg_data/`)
   are re-run every time step 6 fails.
3. A timed-out Kaggle commit generally does not save output, so the trained
   checkpoint is likely lost.

## Proposed restructure — **superseded 2026-09-15**

> These were workarounds for Kaggle's 12h session limit. The move to the department
> GPU server removes that constraint, so the split and the resume logic are no longer
> needed. Kept for the record. See the 2026-09-15 entry.

- **Split into two notebooks.** A: steps 1, 2 and 4 — skip step 3, whose affinity
  matrices feed the AFC figures and are not read by the pruner. Commit; the output
  holds `conv2net_best.pth` and `pg_data/`. B: add A's output as a Dataset input and
  run steps 5-6.
- **Add resume to `pg_pruning`**: read the existing CSV on startup and skip
  combinations already logged. This makes B chainable across sessions.
- **Trim the grid** to 5×5 from `[0.1, 0.3, 0.5, 0.7, 0.9]` — 25 combinations,
  ~12h, one session, and still enough to show Figure 6's "same sparsity, different
  accuracy" effect.

---

# Recommended order of work

1. **Fix `rebuild_fc` (#1) and re-run Conv2Net.** It is a few lines, it decides
   whether the recovery claim means what the paper says it means, and it is cheap to
   test. Nothing else should be measured until it is settled.
2. **Fix the parameter counting (#2)** before any VGG16 sparsity number is reported.
3. **Restore multi-sample PG1 (#3)** and compare against the committed results. If
   ASF-S holds up with a real S-sample PG1, the method is fine and only the
   implementation was wrong.
4. **Match on achieved sparsity (#5)** before building any comparison table.
5. ~~Port the Kaggle notebook to a shell script for the department GPU server~~ —
   done, see `scripts/run_pipeline.sh` in the 2026-09-15 entry.
