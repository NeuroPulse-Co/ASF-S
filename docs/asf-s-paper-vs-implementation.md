# ASF-S: Paper vs Implementation

This doc compares the seed paper ([`ASF-S.md`](../ASF-S.md), the NeurIPS 2026 submission) with the code in this
repo. It covers mainly the CIFAR-100 pipeline in [`cifar100/`](../cifar100/), and also the CIFAR-10 [`min_scripts/`](../min_scripts/)
and [`Bert/`](../Bert/).

Issue numbers (#4, #12, …) refer to the findings in [`cifar100-status.md`](cifar100-status.md).

Last verified against the code: **2026-10-20** (branch `cifar_100`, commit `536f404`).

---

## Summary

- **What matches.** The overall flow matches the paper: output-layer AFC, then PG1 (diffusion map), then the
  correct − incorrect contrast, then PCA, then PGI, then per-layer thresholds, then FC-only fine-tuning. "PG" in
  the code means *Principal Gradient*, as in the paper.
- **Largest deviation: Eq. 12 (PGI projection).** The paper does not define how a conv filter's weight vector
  lines up with output-layer units. The code fills this gap by truncating three vectors from unrelated index
  spaces to a common prefix and multiplying them. The resulting score does not carry the PG1 structure, which is
  consistent with run 02/03, where PGI ranked filters about as well as random.
- **PG1 extraction differs in three ways.**
  - The code row-normalises K into a Markov matrix, which the paper does not do.
  - It calls `eigh` on that non-symmetric matrix.
  - It scales v₂ by λ₂^½ instead of normalising it to unit length.
- **PCA.** The code uses 8 components; the paper uses 4 (≈ 90% variance).
- **Evaluation protocol.**
  - PG prunes by threshold while the baselines prune by ratio.
  - VGG16/Conv6Net use one shared τ instead of per-layer τ.
  - Checkpoints are selected on the test set.
  - Full-model fine-tuning (ASF-S(a)) is missing on CIFAR-100.
- **Setup.** The dataset (CIFAR-100), the models (VGG16 added, LeNet dropped) and some training settings differ
  on purpose. Two baselines, "Sliming" and "SNOWS", are proxies rather than the published methods.

---

## 1. What matches the paper

| Paper | Code |
|---|---|
| Eq. 1–2: unit×unit cosine similarity over S samples | [`pg_ext_cifar100.py:38-50`](../cifar100/pg_ext_cifar100.py#L38-L50), called with `acts.T` at [`:126`](../cifar100/pg_ext_cifar100.py#L126) |
| Eq. 3: `W = exp(-(1-A)²/2σ²)`, symmetrised, `+ εI` | [`pg_ext_cifar100.py:45-49`](../cifar100/pg_ext_cifar100.py#L45-L49) |
| Eq. 5–6: `D = diag(W1)`, `K = D^-α W D^-α`, α = 0.5 | [`pg_ext_cifar100.py:70-75`](../cifar100/pg_ext_cifar100.py#L70-L75) |
| Eq. 9–10: Δg_c = g_c^corr − g_c^incorr, stacked into ΔG | [`pg_pruning_cifar100.py:114-126`](../cifar100/pg_pruning_cifar100.py#L114-L126) |
| Eq. 11: PCA over ΔG | [`pg_pruning_cifar100.py:168-170`](../cifar100/pg_pruning_cifar100.py#L168-L170) |
| Eq. 13: `PGI = N(mean_k \|s_ik\|)`, min-max per layer | [`pg_pruning_cifar100.py:107-108`](../cifar100/pg_pruning_cifar100.py#L107-L108) |
| Eq. 14 / §B.2: one-shot pruning, per-layer τ grid over 0.1–0.9 | Conv2Net 9×9 grid, [`pg_pruning_cifar100.py:189-205`](../cifar100/pg_pruning_cifar100.py#L189-L205) |
| ASF-S(b): pruned layers frozen, only FC fine-tuned | [`pruning_utils.py:151-170`](../cifar100/pruning_utils.py#L151-L170) |
| App. C: Adam, ReduceLROnPlateau (factor 0.1), augmentation, batch 128; Conv2Net/Conv6Net LR 2e-4/3e-4, WD 5e-4 | [`train_cifar100.py`](../cifar100/train_cifar100.py) |

---

## 2. Setup differences (intentional scope changes)

| Item | Paper | Code |
|---|---|---|
| Dataset | CIFAR-10 (Conv2Net, Conv6Net), MNIST (LeNet300_100) | CIFAR-100, with CIFAR-100 normalisation stats ([`config.py:21-22`](../cifar100/config.py#L21-L22)) |
| Models | LeNet300_100, Conv2Net, Conv6Net | Conv2Net, Conv6Net, **VGG16**; no LeNet. Repo is currently set to `vgg16`, while all saved results are for Conv2Net |
| Max epochs | 100 | 200 |
| ReduceLROnPlateau patience | 10 | 5 (plus early stopping on val loss, patience 15; 20 for VGG16) |
| VGG16 optimiser | n/a | SGD 0.05; the cosine schedule is built but **never stepped** ([`train_cifar100.py:136-139`](../cifar100/train_cifar100.py#L136-L139), #8) |
| "Output layer" for PG1 | The output layer | `fc` (logits) for Conv2Net and `fc2` (logits) for Conv6Net, but **`fc1` (hidden)** for VGG16 ([`config.py:61-65`](../cifar100/config.py#L61-L65)) |
| Baselines | Published SVD, SLIMING, SNOWS, HRank, plus L1/L2/top-k/random | HRank, SVD, plus proxies: "Sliming" = singular-value energy × \|BN γ\|, "SNOWS" = mean squared activation over 5 batches |

---

## 3. PG1 extraction ([`pg_ext_cifar100.py`](../cifar100/pg_ext_cifar100.py))

| # | Paper | Code | Effect |
|---|---|---|---|
| 3.1 | Eq. 7: eigendecomposition of **K** | Row-normalises K into a Markov matrix `M = K / rowsum(K)` ([`:76-77`](../cifar100/pg_ext_cifar100.py#L76-L77)) | Different operator. This is the standard diffusion-map step, but the paper does not include it |
| 3.2 | Real eigenvectors of the operator | `torch.linalg.eigh(M)` ([`:81`](../cifar100/pg_ext_cifar100.py#L81)), but M is **not symmetric** after row normalisation | `eigh` reads only one triangle, so the vectors returned are not M's eigenvectors. Using `eigh` on the symmetric K would be both correct and paper-faithful |
| 3.3 | Eq. 8: g = v₂, "normalized to unit length" | `pg1 = v₂ · λ₂^t`, with t = 0.5 ([`:83`](../cifar100/pg_ext_cifar100.py#L83)) | PG1 scale varies per class and split before the subtraction in Eq. 9 |
| 3.4 | No sign handling | Flips g_incorr when `g_corr · g_incorr < 0` ([`pg_pruning_cifar100.py:121-122`](../cifar100/pg_pruning_cifar100.py#L121-L122)) | Necessary, since eigenvector signs are arbitrary, but not in the paper |
| 3.5 | S not stated; split not stated | S ≤ 50 per (class, split) (`PG_SAMPLES`, [`:26`](../cifar100/pg_ext_cifar100.py#L26)), from a balanced subset of the **training** set ([`activation_ext_cifar100.py:116-117`](../cifar100/activation_ext_cifar100.py#L116-L117)) | Run 01 used S = 1 and was degenerate; fixed in run 02 (#15 is still open: train vs test) |
| 3.6 | σ, α, ε not stated | σ = 0.1, α = 0.5, ε = 1e-6 | σ = 0.1 is small enough to risk a disconnected W; hence the `w_offdiag_mean` diagnostic |

---

## 4. PGI scoring (Eq. 12): the main deviation

The paper (Eq. 12) defines

```
s_ik = ⟨ w_i^(ℓ) , u_k ⊙ w̄^(f) ⟩
```

For this to be well defined, `w_i` (a flattened filter), `u_k` (a PCA component of PG1) and `w̄^(f)` (the mean
output-layer weight) must share one index space. The paper never says how they do.

Code ([`pg_pruning_cifar100.py:89-109`](../cifar100/pg_pruning_cifar100.py#L89-L109)):

```python
avg_fc = fc_w.mean(axis=0)                          # mean over FC output rows → length = FC in_features
for comp in comps:                                   # comp length = number of FC output units
    sz   = min(conv.shape[1], avg_fc.shape[0], comp.shape[0])
    proj = conv[:, :sz] @ (comp[:sz] * avg_fc[:sz])
```

For Conv2Net:

| Vector | Indexed by | Length |
|---|---|---|
| `u_k` | FC **output** unit (class) | 100 |
| `w̄^(f)` | FC **input** feature (channel × spatial) | 16384 |
| `w_i` (conv1) | (in_ch, kh, kw) of one filter | 27 |
| `w_i` (conv2) | (in_ch, kh, kw) of one filter | 288 |

- All three vectors are cut to a common prefix (27 entries for conv1) and multiplied **element-wise across
  unrelated indices**. The score is effectively a fixed random projection of each filter's weights.
- PG1 enters only through the first 27 or 288 entries of `u_k`.
- The same `u_k` and `w̄` are reused for every conv layer.

This is #4/#4b in the status doc and the most likely reason PGI performs about like random in runs 02/03.

---

## 5. PCA components

- **Paper (§C.3):** "top 4 components that accounted for ~90% of the variation".
- **Code:** `PCA_COMPS = 8` ([`pg_pruning_cifar100.py:37`](../cifar100/pg_pruning_cifar100.py#L37)).
- The CIFAR-10 [`min_scripts/pg_pruning_conv2.py:52`](../min_scripts/pg_pruning_conv2.py#L52) uses 4.

---

## 6. Pruning and evaluation protocol

| # | Paper | Code |
|---|---|---|
| 6.1 | Table 1 compares methods at a **matched 70% sparsity** | PG uses a threshold, so its sparsity is whatever results. The baselines use an exact per-layer ratio (`keep_mask_from_scores`, [`benchmark_pruning_common.py:128-135`](../cifar100/benchmark_pruning_common.py#L128-L135)). Matched points must be read off a Pareto curve (#5) |
| 6.2 | Layer-specific τ₁…τ_L by grid search | Conv2Net: 2-D grid. **Conv6Net/VGG16: one τ shared by all layers** ([`pg_pruning_cifar100.py:207-213`](../cifar100/pg_pruning_cifar100.py#L207-L213)) |
| 6.3 | Fine-tuning hyperparameters not stated | Adam 1e-4, no scheduler, ≤ 200 epochs, early stopping on val **accuracy** (patience 10/10/15), best weights restored |
| 6.4 | Evaluated on the full test set; checkpoint selection not stated | `val_loader` **is** the test set, so `val_acc` = best test accuracy during fine-tuning, which is optimistically biased |
| 6.5 | ASF-S(a) full-model fine-tuning **and** (b) FC-only | Only (b) exists on CIFAR-100 |
| 6.6 | All convs prunable | Same. FC layers are never pruned. If a layer keeps zero filters, filter 0 is kept ([`pruning_utils.py:19-20`](../cifar100/pruning_utils.py#L19-L20)) |

---

## 7. CIFAR-10 `min_scripts/` (closest to the paper's own runs)

These scripts are the most likely source of the paper's numbers. They differ from both the paper and the fixed
CIFAR-100 code:

- **Eq. 13:** importance is `|mean_k s_ik|` ([`pg_pruning_conv2.py:105`](../min_scripts/pg_pruning_conv2.py#L105)), not `mean_k |s_ik|` (#6).
- **S = 1:** PG1 is built from a single sample (`break  # only 1 sample needed`, [`pg_ext_2.py:83`](../min_scripts/pg_ext_2.py#L83)). The
  cosine matrix is then all ±1 and PG1 is a two-valued sign pattern.
- **`rebuild_fc`:** still has the prefix-copy bug ([`pg_pruning_conv2.py:174-175`](../min_scripts/pg_pruning_conv2.py#L174-L175)), so the classifier is rewired (#1).
- **Fine-tuning:** fine-tunes the **whole model** ([`pg_pruning_conv2.py:190-192`](../min_scripts/pg_pruning_conv2.py#L190-L192)), so its numbers correspond
  to ASF-S(a), not the FC-only (b) the paper highlights (#7).
- PCA uses 4 components, matching the paper.

---

## 8. BERT ([`Bert/main.py`](../Bert/main.py))

| Paper (§5.5, App. D) | Code |
|---|---|
| PG-guided pruning of FFN intermediate neurons in blocks {0, 5, 11} on AG News | Same layers and dataset |
| PGI via Eq. 12 (projection onto `u_k ⊙ w̄^(f)`) | **No weight projection.** Importance is `mean_k \|u_k\|` per neuron ([`:466-469`](../Bert/main.py#L466-L469)). Reasonable here, because the FFN neurons are themselves the PG units |
| PG1 from the output layer | PG1 is computed on each block's **FFN intermediate layer**, not the classifier head |
| Thresholds on normalised PGI | Threshold is a **percentile** of importance ([`:493`](../Bert/main.py#L493)), so it is ratio-like |
| Structured pruning, which reduces FLOPs | Pruned neurons are **zeroed (masked)**, not removed, so no FLOP reduction is realised |
| Head-only fine-tuning | AdamW 2e-5, head-only (`--freeze_encoder`) or full |

---

## 9. Analysis-only artifacts

- **AFC maps.** [`cosine_sim_ext_cifar100.py`](../cifar100/cosine_sim_ext_cifar100.py) builds **sample×sample** raw cosine matrices with no
  Gaussian kernel (#12). Paper Fig. 3 shows unit×unit AFC maps through Eq. 3. The pruner never reads these maps,
  so this affects figures only.
- **Activations.** Only the `PG_FC_LAYER` activations feed PG1. The activations of the other hooked layers
  are not used for scoring.

---

## 10. Suggested fixes

Ordered by expected impact. None of these are applied yet.

1. **Define Eq. 12 dimensionally.** For example, score each conv filter through the FC input columns it feeds
   (`keep_idx * hw + arange(hw)`, as in `rebuild_fc`) and project those FC weights onto `u_k`. This changes the
   method, so agree on it with the supervisor first.
2. **PG1 extraction.**
   - Run `eigh` on the symmetric K (or `eig` on M).
   - Unit-normalise v₂ and drop the λ^t scaling.
3. **PCA.** Set `PCA_COMPS = 4`, or choose K by ≥ 90% explained variance.
4. **Validation split.** Hold out a validation split from the training set for early stopping, and report test
   accuracy once.
5. **Ratio-matched PG mode.** Add top-k by PGI, so comparisons with the baselines are like-for-like.
6. **Per-layer τ for Conv6Net/VGG16.** Use per-layer thresholds there, with random or coordinate search to keep
   the grid tractable.
7. **ASF-S(a) on CIFAR-100.** Add a full-model fine-tuning option.
