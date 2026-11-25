# Run 02 — Conv2Net / CIFAR-100, fixes #1–#3 applied (S = 50)

The first run whose numbers mean what their column names say. Code = the `cifar_100` branch with the
minimal fixes from 2026-09-28 (see
[`docs/cifar100-status.md`](../../../../docs/cifar100-status.md#code-changes-for-run-02)):
`rebuild_fc` copies the kept channels' columns, parameters are counted in full, and PG1 uses S = 50.
Everything else, including the Eq. 12 formula and the τ grid, is unchanged from the fork.

| | |
|---|---|
| Dates | 2026-09-28 13:14 → 2026-09-29 09:27 |
| Host / env | same as run 01: RTX 2080, torch 2.5.1+cu121, Python 3.10 |
| Checkpoint | **reused from run 01** (`conv2net_best.pth`, 52.52% in training under AMP; 52.36% in FP32) |
| Steps | 2 (activations, 8.8 GB), 4 (PG1, 10 s), 6 (81 combinations, 1024 min) |
| Files here | `results/pg_conv2net_cifar100.csv` (81 rows), `pg_data/pg1_summary.csv`; the `.npy` files are gitignored |

## PG1 is now non-degenerate

| | Run 01 (S = 1) | Run 02 (S = 50) |
|---|---|---|
| Constant `incorrect` vectors | 26 / 100 | **0** |
| Distinct values per vector (median, correct / incorrect) | 2 / 2 | 38 / 99 |
| W off-diagonal entries > 0.01 (mean) | — | 87% / 88%, so σ = 0.1 is not a problem |
| PCA over ΔG: PC1 / top-8 | 0.050 / 0.309 | **0.388 / 0.634** |

## Pruning results (FC-only fine-tune)

The best `val_acc` at each conv2 keep level. At every level except the last, this row keeps conv1 almost intact (τ₁ = 0.1 → 30/32 kept).

| conv2 kept | Params ↓ | FLOPs ↓ | `pre_acc` (no FT) | `val_acc` | Drop vs 52.52 |
|---|---|---|---|---|---|
| 57/64 | 11.0% | 15.5% | 33.67 | **51.12** | −1.4 |
| 47/64 | 26.6% | 29.6% | 21.38 | 49.18 | −3.3 |
| 31/64 | 51.6% | 52.1% | 13.64 | 45.75 | −6.8 |
| 20/64 | **68.7%** | 67.5% | 5.90 | **40.46** | **−12.1** |
| 9/64 | 85.9% | 83.0% | 2.79 | 29.59 | −22.9 |
| 5/64 | 92.1% | 88.6% | 1.40 | 22.19 | −30.3 |
| 3/64 | 95.3% | 91.4% | 1.85 | 16.23 | −36.3 |
| 2/64 | 96.9% | 98.8% | 0.99 | 13.17 | −39.4 |

Run-to-run noise: nine pairs of combinations produced identical kept counts (τ₂ = 0.8 and 0.9 both keep
2 filters). Their `val_acc` differs by 0.29 on average and 0.73 at most. Treat differences under ~1 point as noise.

## What this run shows

1. **The fixes work.** `pre_acc` is now a real measurement: 33.67% at the lightest setting, where every
   fork row sat at chance (≈1%). The parameter percentages include the conv layers.
2. **Final accuracy did not change.** At matched parameter reduction, run 02's `val_acc` is within
   about 0–2 points of the fork's results (fork, S = 1 and broken `rebuild_fc`: ≈42.0% at 68.7%, against 40.46%
   here). Retraining the FC recovers roughly the same accuracy whether or not PG1 was degenerate. So the final
   `val_acc` is not sensitive to PGI quality.
3. **The paper's headline claim does not reproduce on CIFAR-100.** "FC-only fine-tuning maintains 100%
   of baseline at up to 70% sparsity" (the paper's CNN claim, made on CIFAR-10): here, 68.7% sparsity costs
   12.1 points. Only the lightest setting (11%) comes within 1.4 points.
4. **Pruning is far from function-preserving before fine-tuning.** Removing 2 conv1 and 7 conv2 filters
   drops accuracy from 52.4% to 33.7% before any retraining.
5. **"Same sparsity, different accuracy" (the paper's Fig. 6) is largely a parameter-count artefact.**
   The FC layer holds 98.8% of Conv2Net's parameters, so the parameter percentage depends almost entirely on
   conv2. Pruning conv1 barely moves it but costs accuracy (at 20 conv2 kept: 40.5% → 29.0% as conv1
   goes 30 → 4) and saves FLOPs. FLOPs, not parameters, is the axis on which conv1 pruning should be judged.

## Does PG's choice of filters beat random or L1 at the same counts?

Points 2 and 4 make this the deciding question. It was checked on 2026-09-29 on the server GPU, with the
same checkpoint and no fine-tuning.
- **PG:** the filters step 6 chose. The PGI was rebuilt from this run's `pg_data`, and it reproduced
  all 81 kept counts exactly.
- **L1:** the highest-L1-norm filters.
- **Random:** 10 random draws.

Every method keeps the same number of filters per layer.

| τ | kept (conv1/conv2) | PG | L1 | random, mean ± sd (range) | random ≥ PG |
|---|---|---|---|---|---|
| (0.1, 0.1) | 30/57 | 33.67 | **42.10** | 28.66 ± 5.77 (17.69–38.28) | 2/10 |
| (0.1, 0.2) | 30/47 | 21.38 | **35.27** | 19.08 ± 5.54 (6.69–26.50) | 5/10 |
| (0.1, 0.3) | 30/31 | 13.64 | **20.98** | 9.48 ± 2.92 (2.77–13.47) | 0/10 |
| (0.1, 0.4) | 30/20 | 5.90 | **16.10** | 5.63 ± 2.47 (1.35–10.20) | 4/10 |
| (0.1, 0.5) | 30/9 | 2.79 | **4.41** | 3.79 ± 1.17 (2.14–5.33) | 8/10 |
| (0.3, 0.1) | 24/57 | **12.56** | 10.67 | 8.11 ± 3.76 (2.78–14.24) | 2/10 |
| (0.5, 0.1) | 17/57 | 3.65 | **5.58** | 3.77 ± 1.41 (2.68–6.48) | 3/10 |
| (0.3, 0.3) | 24/31 | 3.49 | **6.01** | 4.20 ± 1.92 (2.53–8.68) | 6/10 |

- **PG is only slightly better than a random choice.** It averages +1.8 points over the random mean,
  always within about one standard deviation, and 30 of the 80 random draws matched or beat it.
- **Plain L1-norm beats PG in 7 of 8 rows**, often by a wide margin (42.1 vs 33.7 at the lightest
  setting; 16.1 vs 5.9 at 69% sparsity).

**After FC fine-tuning the picture is the same.** At a 70% per-layer ratio the fork's baselines kept
10/19 filters. PG's nearest grid point keeps 10/20, one extra filter at the same FLOPs (87.2% vs 87.8%):

| Method (fork's results, except PG) | kept | `val_acc` after FC fine-tune |
|---|---|---|
| Sliming | 10/19 | **36.16** |
| SVD | 10/19 | 35.92 |
| SNOWS | 10/19 | 33.69 |
| **ASF-S / PG (run 02)** | 10/20 | **32.00** |
| HRank (ratio 0.2) | 25/51 | 47.58, vs PG 26/47: 47.30 |

The baseline rows come from the fork, so they used the older checkpoint (51.93%) and the old
`rebuild_fc`. Because the FC is retrained anyway, that has little effect on `val_acc` (point 2 above).
The 2–4 point gaps are well above the ~0.7-point noise. Re-running the baselines on the fixed code
would make this comparison airtight.

**Conclusion for run 02:** on CIFAR-100 Conv2Net, ASF-S's PGI ranks filters about as well as chance and
worse than L1-norm, both before and after fine-tuning. Its accuracy after fine-tuning comes from
retraining the FC, not from PGI. The most likely cause is Eq. 12 as implemented (#4/#4b). Each filter's
score is its first 27 (conv1) or 100 (conv2) weights, dotted against unrelated entries of the class-indexed
`u_k` and the feature-indexed `w̄`, which amounts to a fixed random projection of the weights. The S fix made
PG1 meaningful, but the step that turns PG1 into filter scores discards that structure.
