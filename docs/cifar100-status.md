# ASF-S on CIFAR-100 — status and findings

Where the CIFAR-100 replication stands, what is wrong with the pipeline, and what to do next.
The doc is ordered by topic. The dated evidence behind each claim is in
[`engineering-log.md`](engineering-log.md).

Last verified against the code and data: **2026-09-29** (run 02 results).

---

## Summary

- **Goal:** reproduce the paper's ASF-S pruning results on CIFAR-100 (Conv2Net first, then
  Conv6Net and VGG16) starting from the fork's code.
- **Status (2026-09-29):** run 02 finished, with fixes #1–#3 and all 81 τ combinations. **Headline:** the
  fixes work (`pre_acc` is real, PG1 is non-degenerate), but ASF-S does **not** reproduce on CIFAR-100
  Conv2Net. At 68.7% sparsity, FC-only fine-tuning reaches 40.46% (−12.1 pts). PGI picks filters only
  slightly better than random and worse than plain L1-norm, both before and after fine-tuning.
  → [run 02 README](../cifar100/pg_project_output/runs/run02_2026-09-28_rtx2080_S50_fixes/README.md)
- **Most likely cause:** Eq. 12 as implemented (#4/#4b) reduces each filter's score to an arbitrary
  projection of its weights, so the (now meaningful) PG1 structure never reaches the pruning decision.
- **Top blocker, the "S problem":** PG1 is computed from **one image** per (class, split), not from
  S samples as the paper defines. With S = 1 the maths collapses: every PG1 vector is a two-valued
  sign pattern that says, in effect, "which logits were positive". → [The S problem](#the-s-problem)
- **Two further blockers** make the reported numbers mean something other than what they claim:
  `rebuild_fc` rewires the classifier (so `pre_acc` is meaningless), and the parameter counts leave out the
  conv layers. → [Other findings](#other-findings)
- **Fixing S alone may not be enough.** With σ = 0.1 the affinity matrix may still be almost empty.
  Check this on real activations before paying for the ~50 GPU-hour grid. → [Risks after the S fix](#risks-that-remain-after-fixing-s)
- **Code fixed 2026-09-28 for run 02:** #1 `rebuild_fc`, #2 param counts, #3 S = 50, plus PG1 sign
  alignment and the dump size. Nothing else changed, on purpose. → [Code changes for run 02](#code-changes-for-run-02)

---

## Where the results live

All under `cifar100/pg_project_output/`:

| Folder | What | Rule |
|---|---|---|
| [`original_fork/`](../cifar100/pg_project_output/original_fork/README.md) | The fork's CSVs at commit `6e8b82a`, i.e. what we are reproducing | read-only, never write here |
| [`runs/run01_2026-09-16_rtx2080_S1/`](../cifar100/pg_project_output/runs/run01_2026-09-16_rtx2080_S1/README.md) | Our run 01: unmodified code, stopped after step 4 | archived evidence, not results |
| [`runs/run02_2026-09-28_rtx2080_S50_fixes/`](../cifar100/pg_project_output/runs/run02_2026-09-28_rtx2080_S50_fixes/README.md) | Our run 02: fixes #1–#3, steps 2, 4, 6 complete (81 rows) | archived; the first valid PG results |
| `results/`, `pg_data/`, `activations/`, `affinity_matrices/` | **Live slot.** The scripts write here (`config.py`). On the server these still hold run 02's outputs; clear them before run 03 | — |
| `trained_models/` | Live checkpoint. It holds run 01's `conv2net_best.pth` so run 02 skips training | gitignored |

When run 02 finishes, move its outputs to `runs/run02_<date>_<what-changed>/` with a README in the
same format as run 01. Tidying the code-level layout (for example, `config.py` writing straight into
a run folder) is deferred until a run succeeds.

---

## Code changes for run 02

Made 2026-09-28. Kept deliberately minimal, so that run 02 differs from run 01 only in what is
needed for the numbers to mean what they claim.

| Fix | Where | What changed |
|---|---|---|
| #1 `rebuild_fc` | [`pruning_utils.py`](../cifar100/pruning_utils.py) `rebuild_fc(model, fc_attr, keep_idx)` | copies the **kept channels'** FC columns (`c·H·W + p`), not a prefix. Raises if the shapes disagree. All callers pass the last conv's kept `idx`: PG, HRank, and SVD/Sliming/SNOWS via `benchmark_pruning_common.py` |
| #2 param counts | `count_params()` in `pruning_utils.py`, used in the same three scripts | counts every parameter for both the base and pruned models. `ptflops` is still used for FLOPs only |
| #3 S = 1 | [`pg_ext_cifar100.py`](../cifar100/pg_ext_cifar100.py) `process()` | stacks up to `PG_SAMPLES = 50` samples per (class, split) → Eq. 1 as written. File order is seeded. S < 2 is logged as `few_samples` and skipped |
| sign alignment | `load_pg_diffs` in [`pg_pruning_cifar100.py`](../cifar100/pg_pruning_cifar100.py) | an eigenvector's sign is arbitrary, so `g_incorr` is flipped if `g_corr · g_incorr < 0` before Eq. 9. **Our implementation choice**; the paper does not address it |
| #13 dump size | [`activation_ext_cifar100.py`](../cifar100/activation_ext_cifar100.py) | saves `inputs[i].clone()`, so one image is stored instead of the whole batch |

`pg1_summary.csv` gains diagnostic columns (they don't change any result):
`n_samples`, `pg1_distinct` (number of distinct PG1 values; it was 2 at S = 1), `w_offdiag_mean` and `w_frac_gt_0.01`.
The last two are how connected W is at σ = 0.1. If `w_frac_gt_0.01` is near 0, PG1 is degenerate
again and σ needs revisiting before step 6.

**Deliberately not changed:** the Eq. 12 formula (#4/#4b, a question for the authors); threshold
selection and the τ grid (#5); σ, α, t, `PCA_COMPS`; the `eigh` call; the VGG16 schedule (#8);
`min_scripts/`; the AFC-map script (#12).

**How it was verified (CPU, synthetic):** pruning random channel subsets of Conv2Net, Conv6Net and
VGG16 now gives logits identical (max |Δ| ≈ 1e-7) to the unpruned model with those channels
silenced. The old code fails the same check. `count_params` matches a hand count with layers frozen
(Conv2Net base = 1,658,084). The modified `pg_ext` gives 448–512 distinct PG1 values at S = 50 and
flags S = 1 as `few_samples`. Not yet run on real data.

---

## The paper in one page

*Artificial Structure Function Search* (NeurIPS 2026 submission). The idea: a trained network's
output layer has a "functional connectivity" (AFC), meaning how its units co-activate across
images. Prune the hidden filters that contribute least to it, and the network should recover
accuracy after retraining **only the output layer**.

The pipeline, with the code that implements each step (Conv2Net, CIFAR-100):

| # | Paper | What it does | Code |
|---|---|---|---|
| 1 | Eq. 1 | `X ∈ ℝ^{N×S}`: each unit's activations across **S samples** is its "time series" | activations saved by [`activation_ext_cifar100.py`](../cifar100/activation_ext_cifar100.py); run 01 loaded **S = 1**, now up to 50 in `process()` of [`pg_ext_cifar100.py`](../cifar100/pg_ext_cifar100.py) |
| 2 | Eq. 2 | Cosine similarity between units: `A_ij` | [`pg_ext_cifar100.py:44-45`](../cifar100/pg_ext_cifar100.py#L44-L45) |
| 3 | Eq. 3 | Gaussian kernel `W_ij = exp(−(1−A_ij)²/2σ²)`, σ = 0.1 | [`:46`](../cifar100/pg_ext_cifar100.py#L46) |
| 4 | Eq. 5–6 | Degree `D`, diffusion operator `K = D^−α W D^−α`, α = 0.5 | [`:70-73`](../cifar100/pg_ext_cifar100.py#L70-L73), then row-normalised to a Markov matrix at `:74-75` (not in the paper) |
| 5 | Eq. 7–8 | PG1 = second eigenvector of K | [`:81-83`](../cifar100/pg_ext_cifar100.py#L81-L83), scaled by `λ₂^0.5`, so not unit length as Eq. 8 says |
| 6 | Eq. 9–10 | Per class: `Δg_c = g_c^correct − g_c^incorrect`; stack to ΔG (C × units) | [`pg_pruning_cifar100.py:114-125`](../cifar100/pg_pruning_cifar100.py#L114-L125) |
| 7 | Eq. 11 | PCA on ΔG → top K components `u_k` | [`:169`](../cifar100/pg_pruning_cifar100.py#L169), K = 8 (paper §C.3 uses 4) |
| 8 | Eq. 12 | Score `s_{i,k} = ⟨w_i, u_k ⊙ w̄^(f)⟩` for each filter i | [`:98-104`](../cifar100/pg_pruning_cifar100.py#L98-L104) |
| 9 | Eq. 13 | `PGI_i = minmax( mean_k |s_{i,k}| )` | [`:107-108`](../cifar100/pg_pruning_cifar100.py#L107-L108) |
| 10 | §4 | Prune filters with PGI < τ (per layer), grid-search τ | [`:190-197`](../cifar100/pg_pruning_cifar100.py#L190-L197) |
| 11 | §5 | Freeze convs, rebuild and retrain the FC only | [`:221-234`](../cifar100/pg_pruning_cifar100.py#L221-L234) → `rebuild_fc` |

### Glossary

| Term | Meaning |
|---|---|
| **AFC** | Artificial Functional Connectivity: the unit-by-unit similarity structure of a layer (Eq. 2–3) |
| **PG1** | First principal gradient: the 2nd eigenvector of the diffusion operator. One value per unit |
| **PGI** | Principal Gradient Importance: the per-filter pruning score (Eq. 13) |
| **S** (capital) | Number of **samples** (images) per unit in Eq. 1. **This is the S problem.** |
| **s_{i,k}** (small) | Eq. 12's projection score for filter i on PCA component k. A separate issue, finding #4 |
| σ | Gaussian kernel width in Eq. 3 (code: 0.1) |
| τ | Pruning threshold on the min-max-normalised PGI. It is **not** a sparsity ratio |

---

## Repo map

| Step | Script | Notes |
|---|---|---|
| 1 | `cifar100/train_cifar100.py` | ~64 min on the RTX 2080 |
| 2 | `cifar100/activation_ext_cifar100.py` | ~39 GB for Conv2Net. Saves 100 correct + 100 incorrect per class, from the **training** set |
| 3 | `cifar100/cosine_sim_ext_cifar100.py` | AFC maps for figures only; the pruner does not read them |
| 4 | `cifar100/pg_ext_cifar100.py` | PG1 per (class, split) |
| 5 | `cifar100/hrank_cifar100.py` | HRank baseline |
| 6 | `cifar100/pg_pruning_cifar100.py` | ASF-S pruning + FC fine-tune. 81 combinations ≈ 50 h for Conv2Net |
| — | `svd_ / sliming_ / snows_pruning_cifar100.py` | other baselines via `benchmark_pruning_common.py` |
| — | [`scripts/run_pipeline.sh`](../scripts/run_pipeline.sh) | server runner: `--setup`, `--steps 1,2,4`. Picks the torch wheel from the driver version |
| — | [`scripts/select_model.py`](../scripts/select_model.py) | switches the `MODELS` dict in every script to one model; `--check` shows the current one |
| — | `min_scripts/` | the older CIFAR-10/MNIST scripts behind the paper's figures. Cannot currently run (no `models.py`) |

Working server env: torch 2.5.1+cu121, Python 3.10, RTX 2080 (driver 535, CUDA 12.2).

---

## The S problem

### What the paper says

Eq. 1 defines `X^(ℓ) ∈ ℝ^{N_ℓ × S}`. Each of the N units gets a vector of its activations over S
images, and §3 calls this "the analogue of a neural time series". Cosine similarity between two
units' vectors (Eq. 2) measures whether they **co-fluctuate across images**. That similarity is the
functional connectivity everything else is built on.

### What the code did (run 01; fixed 2026-09-28)

`process()` in `pg_ext_cifar100.py` (run 01 code, last changed in `c485447`) shuffled the class's files,
took the **first** one that matched the split, and `break`s:

```python
acts = d["activations"][layer_name].flatten().numpy()[None, :]   # shape (1, units)
break
...
A = cosine_affinity_gpu(acts.T)       # (units, 1): each unit has ONE number
```

So **S = 1**. The CIFAR-10 original, [`min_scripts/pg_ext_2.py:83`](../min_scripts/pg_ext_2.py#L83),
does the same (`break  # only 1 sample needed`), even though it declares `max_samples = 50` at
line 25 and never uses it. Step 2 saves up to 100 samples per split, so the data needed for S = 50
was already there; run 01 simply never read it. The CIFAR-100 script now stacks up to 50 (see [Code changes for run 02](#code-changes-for-run-02)).

### Why S = 1 breaks the maths

1. Each unit is a single scalar, and `F.normalize` of a scalar is **±1** (or 0).
2. Cosine between two ±1 scalars is **±1**. So A has only two values: same sign or opposite sign.
3. Eq. 3 with σ = 0.1: `W = exp(0) = 1` for same sign, `exp(−200) ≈ 0` for opposite sign.
4. W is therefore a **two-block matrix**, and its second eigenvector just labels the two blocks.

**PG1 at S = 1 is the sign pattern of one image's activations.** Nothing about co-fluctuation
survives. A numpy re-implementation of the code on random logits gives exactly 2 distinct PG1
values, split by sign.

### What the saved data shows (run 01, Conv2Net `fc`, 100 units)

| Check | `correct` | `incorrect` |
|---|---|---|
| Vectors with 2 distinct values | 96 | 74 |
| Vectors that are **constant** (all units the same, so no information) | 4 | **26** |
| Typical split | 1/99 … 8/92 | same |

- **The true-class unit is in the minority block in 96/96 two-valued `correct` vectors.** So
  "correct" PG1 amounts to "the true class plus the few other classes whose logit was positive".
  It is a top-prediction indicator, not a connectivity structure.
- Some classes have correct ≈ incorrect to 6 significant figures (class 1: Δg ≈ 4e-7).
- **ΔG (100 × 100) has rank 99 and a flat spectrum:** PC1 explains 5.0%, PCA(8) explains **30.9%**, and 75
  components are needed for 99%. Eq. 11 is supposed to find "the dominant, class-shared axes". There
  is none, so the `u_k` fed into Eq. 12 are close to arbitrary.
  (The paper's "4 components ≈ 90%" is CIFAR-10 with a 10-unit output. That is not a contradiction,
  but it does not carry over to 100 units.)

### What this does and does not prove

- **Proven:** the PG1 vectors are degenerate, and the PGI scores are built on them.
- **Not proven:** that ASF-S does not work. The fork's PG rows still reach 50.03% at light pruning.
  An almost arbitrary score can still beat random pruning, and all the fork numbers also pass through
  the `rebuild_fc` bug. Whether the *method* works is what run 02 with S = 50 answers.
- A null model does not support calling it "pure noise": random two-valued rows show *more*
  concentration (PC1 ≈ 0.50) because of a shared offset.

### Reproduce (no GPU, a few seconds)

```python
import numpy as np, glob, os
base = 'cifar100/pg_project_output/runs/run01_2026-09-16_rtx2080_S1/pg_data/conv2net'
def load(split):
    fs = sorted(glob.glob(f'{base}/{split}/pg1_data/*_fc_pg1.npy'),
                key=lambda p: int(os.path.basename(p).split('_')[0]))
    return np.stack([np.load(f) for f in fs])
C, I = load('correct'), load('incorrect')
print([len(np.unique(np.round(r, 6))) for r in C][:10])       # distinct values per class
print(sum(len(np.unique(np.round(r, 6))) == 1 for r in I))     # constant vectors -> 26
dG = C - I; X = dG - dG.mean(0)
ev = np.linalg.svd(X, compute_uv=False)**2; ev /= ev.sum()
print(ev[:8].round(4), ev[:8].sum())                           # -> 0.309
```

(The `.npy` files are gitignored. They are on the machines run 01 was copied to.)

---

## Risks that remain after fixing S

These need checking **on real activations** with S = 50 before running the grid. The numbers
below come from synthetic data.

1. **σ = 0.1 may leave W almost empty.** *Resolved by run 02: on real logits 87–88% of the off-diagonal W entries exceed 0.01 (the minimum across classes is 61%), so this risk did not occur.* The original concern: `exp(−(1−cos)²/0.02)` is only non-negligible when
   cos ≳ 0.7. For weakly correlated units (cos ≈ 0) it gives ≈ 3e-6. In a synthetic S = 50 test, only
   **0.02%** of off-diagonal W entries exceeded 0.01. A nearly disconnected graph gives a PG1 that
   isolates one unit, which is another degenerate result. Logits of same-class images may well be more
   correlated than this, so *measure* the off-diagonal distribution of W first. Consider a larger
   σ, or the paper's alternative of Pearson correlation.
2. **`eigh` on a non-symmetric matrix (minor).** The code row-normalises K into M
   ([`:74-75`](../cifar100/pg_ext_cifar100.py#L74-L75)) and then calls `torch.linalg.eigh`, which
   assumes symmetry and reads only one triangle. In the synthetic test the result still matched the
   true eigenvector (|cos| = 0.99). The clean fix is to eigendecompose the symmetric
   `D^-½ K D^-½` and rescale.

---

## Other findings

Numbers #1–#11 match [`engineering-log.md`](engineering-log.md#severity-summary). Items marked
**new** were found in the 2026-09-28 re-check. *Likely* means inferred, not yet measured.

| # | Finding | Effect | Status |
|---|---|---|---|
| 1 | `rebuild_fc` copied the first *m* FC columns, not the kept channels' columns | every `pre_acc` ≈ 1%, so meaningless. The "FC-only recovery" becomes a linear probe on a rewired FC, not evidence that AFC was preserved | **fixed 2026-09-28** |
| 2 | Pruned params measured after freezing the convs, so `ptflops` counted only the FC | sparsity % inflated. ≤1.2% error for Conv2Net, severe for VGG16. FLOPs are fine | **fixed 2026-09-28** (verified: `params` = 58·256·100+100) |
| 3 | S = 1 | see [above](#the-s-problem) | **fixed 2026-09-28** (S = 50). The σ risk remains |
| 4 | Eq. 12 lengths differ, so the code truncates to the shortest prefix ([`:103-104`](../cifar100/pg_pruning_cifar100.py#L103-L104)) | Conv2Net/C100: `sz` = 27 (conv1), 100 (conv2) | verified |
| 4b | **new:** Eq. 12 also mixes *index spaces*. `u_k` is indexed by **class** (100), `w̄ = fc_w.mean(0)` by **FC input feature** (16 384), and `w_i` by (in_ch, kh, kw) | the element-wise product pairs unrelated entries, which is worse than truncation. The paper never defines the dimensions | verified |
| 5 | PG prunes by threshold τ, baselines by ratio | PG τ 0.7 ≠ 70% sparsity (VGG16: 99.5%). Compare at *achieved* sparsity | verified |
| 6 | `min_scripts` uses `|mean(s)|`; Eq. 13 says `mean(|s|)` ([`pg_pruning_conv2.py:105`](../min_scripts/pg_pruning_conv2.py#L105)) | affects CIFAR-10 paper figures only | verified |
| 7 | `min_scripts` fine-tunes the whole model ([`:192`](../min_scripts/pg_pruning_conv2.py#L192)) | those are ASF-S(a) numbers, not (b) | verified |
| 8 | VGG16 cosine LR schedule built but never stepped ([`train_cifar100.py:138-139`](../cifar100/train_cifar100.py#L138-L139)) | VGG16 baseline weak: best 58.17% @ 88, final 55.99%. The log's "57.14%" does not match the CSV | verified |
| 9 | `svd_conv2net` / `svd_conv6net` CSV headers corrupted | readable by hand, see the fork README | verified |
| 10 | Activation extraction rescans the dataset once per class ([`activation_ext_cifar100.py:90`](../cifar100/activation_ext_cifar100.py#L90)) | runtime only | verified |
| 11 | Root `README.md` is stale | onboarding only | verified |
| 12 | **new:** the CIFAR-100 AFC maps are **sample × sample** raw cosine with no Gaussian kernel ([`cosine_sim_ext_cifar100.py:77-79`](../cifar100/cosine_sim_ext_cifar100.py#L77-L79)). The CIFAR-10 original computed both unit × unit and sample × sample | any CIFAR-100 "AFC map" figure would not be Eq. 2–3. The pruner is unaffected | verified |
| 13 | **new:** the dump is ~2 MB/file, not ~460 KB, *likely* because `"input": inputs[i].cpu()` ([`activation_ext_cifar100.py:67`](../cifar100/activation_ext_cifar100.py#L67)) is a view of the CPU batch, so `torch.save` writes the whole batch (128·3·32·32·4 B ≈ 1.57 MB). The activations are copied compactly from the GPU, so they are not the cause | `.clone()` → ~4× smaller | **fixed 2026-09-28**; measure the size of one file in run 02 |
| 14 | **new:** fork `pg_conv2net` rows 3 and 7 both say τ = (0.1, 0.7) but keep 43 vs 2 conv2 filters | row 3 is probably τ₂ = 0.3 with the wrong label | verified anomaly |
| 15 | **new:** PG1 samples come from the **training** set (the first 1000 per class) | "incorrect" training images are few and atypical, which helps explain the 26 constant vectors | verified |

---

## Run history and costs

| When | Where | What happened |
|---|---|---|
| 2026-09-13 | Kaggle, 2× T4 | Steps 1–5 done; step 6 timed out at 12 h (15 of 81 combinations); all output lost |
| 2026-09-15 | → department server | Moved for no session limit; `run_pipeline.sh` added; torch/driver mismatch fixed in `--setup` |
| 2026-09-16 | server | Step 1 ✅ 64 min. Step 2 ❌ `/home` (a shared 1.6 TB volume) hit 100% at class 66 |
| 2026-09-18 | server | Step 2 ✅ 13 min (the storage fix was not recorded). Step 4 ✅ 2.7 s. PG1 collapse confirmed |

Measured costs (RTX 2080): 23 s per training epoch (CPU/DataLoader bound, `num_workers=4`), about 37 min per
pruning combination, about 50 h for the 81-combination Conv2Net grid, ~39 GB activation dump (~4× the estimate).

---

## Next steps (after run 02)

Run 02 finished steps 1–3 of the old list (the fixes) and answered the S / σ question: W is well
connected at σ = 0.1. What remains:

1. **Re-run the baselines on the fixed code** (HRank, SVD, Sliming, SNOWS, plus L1 and random at PG's
   kept counts, with FC fine-tuning). The fork's rows used the old checkpoint and the old `rebuild_fc`;
   re-running them makes the "PG is worse than L1/Sliming" comparison airtight. Cost: about 35 min per row.
2. **Take Eq. 12 (#4/#4b) to the paper authors.** It is the most likely reason PGI ≈ random. Without a
   defined mapping from the class-indexed `u_k` to each filter, the method cannot use the PG1 structure
   it computes. Any redefinition is a change to the method, not a bug fix, so it needs their agreement.
3. **Report on FLOPs as well as parameters.** On Conv2Net, parameter % depends almost entirely on conv2
   (the FC holds 98.8% of the parameters), so conv1 pruning shows up only in FLOPs.
4. Only then: Conv6Net / VGG16 (for VGG16, fix #8's LR schedule before retraining).

## Paper vs code: raise these with the authors before the camera-ready

- Eq. 1 needs S samples; the code uses 1 (both CIFAR-10 and CIFAR-100).
- Eq. 12 has no well-defined dimensions (#4, #4b).
- Eq. 13 is `mean(|s|)`; the CIFAR-10 figures used `|mean(s)|` (#6).
- Eq. 8 says PG1 is unit length; the code scales it by `λ₂^t` and uses a row-normalised operator.
- §C.3 uses 4 PCA components; the CIFAR-100 code uses 8.
- Which script produced which figure: `min_scripts` is full fine-tuning (#7).
- The dashed "post-prune" curves (Fig. 7) cannot come from this code path while #1 stands.
