# Run 03 — baselines on the fixed code (Conv2Net / CIFAR-100)

SVD, Sliming, SNOWS and HRank re-run on the **same checkpoint** as run 02 (52.52%) and the **same fixed
code** (`rebuild_fc` copies the kept channels, parameters counted in full). This is the first
like-for-like comparison with ASF-S/PG: every method uses FC-only fine-tuning with the convs frozen.

| | |
|---|---|
| Date | 2026-09-29, 12:07 → 13:59 |
| Commands | `run_pipeline.sh --model conv2net --steps svd,sliming,snows --ratios 0.7`, then `--steps 5` |
| Host / env | see `RUN_INFO.md`. Its "uncommitted code changes" flag is the `select_model.py` model-selection edits, not algorithm changes |
| Selection | exact per-layer ratio: SVD, Sliming and SNOWS at 0.7; HRank at its built-in 0.2–0.9 |

## Results

| Method | Ratio | Kept (conv1/conv2) | Params ↓ | FLOPs ↓ | `pre_acc` | `val_acc` | Fork's result |
|---|---|---|---|---|---|---|---|
| SVD | 0.7 | 10/19 | 70.5% | 87.8% | 1.15 | 33.99 | 35.92 |
| **Sliming** | 0.7 | 10/19 | 70.5% | 87.8% | 1.75 | **37.02** | 36.16 |
| SNOWS | 0.7 | 10/19 | 70.5% | 87.8% | 1.41 | 32.34 | 33.69 |
| HRank | 0.2 | 25/51 | 20.5% | 35.4% | 10.04 | 46.88 | 47.58 |
| HRank | 0.3 | 22/44 | 31.5% | 49.7% | 7.44 | 45.63 | — |
| HRank | 0.5 | 16/32 | 50.3% | 71.5% | 2.23 | 42.23 | — |
| HRank | 0.6 | 12/25 | 61.2% | 82.0% | 1.71 | 38.55 | — |
| HRank | 0.8 | 6/12 | 81.4% | 94.4% | 2.32 | 29.21 | — |
| HRank | 0.9 | 3/6 | 90.7% | 97.9% | 1.26 | 19.68 | — |

**The fork's baseline numbers hold up.** They are within −1.9 to +0.9 points of the fork's rows (which
used the old checkpoint and the old `rebuild_fc`). This confirms run 02's finding that, once the FC is
retrained, the `rebuild_fc` bug hardly affects the final `val_acc`.

## Against ASF-S / PG (run 02)

PG's 81 grid points and the baselines' fixed ratios land on different kept counts, so the comparison uses
Pareto dominance. A method *dominates* a point when it is at least as compressed on **both** parameters and
FLOPs and at least as accurate. 46 of PG's 81 rows are on its own Pareto front.

| Baseline | Verdict vs PG's front | Detail |
|---|---|---|
| HRank 0.2 | **PG better** | PG 26/47 (26.7% / 38.1%) gets 47.30 vs 46.88 while being more compressed. The margin is within noise |
| HRank 0.3 | **PG better** | PG 24/47 (26.8% / 42.3%) gets 46.66 vs 45.63 |
| HRank 0.5 | baseline better | dominates PG 12/57 (41.7) and 10/57 (41.6) |
| HRank 0.6 | baseline better | dominates PG 12/31 (52% / 78%, 36.8) |
| HRank 0.8 | baseline better | dominates PG 4/20 (69% / 93%, 29.0) |
| HRank 0.9 | neither | no PG point either side |
| **Sliming 0.7** | **baseline clearly better** | dominates **5** PG front points, including PG 10/20 (69% / 87%, 32.0) by **+5.0 pts** |
| SVD 0.7 | baseline better | dominates PG 10/20 by +2.0 |
| SNOWS 0.7 | ≈ | dominates PG 10/20 by only +0.34, within the 0.73-point noise floor |

The noise floor comes from run 02: combinations with identical kept counts differ by up to 0.73 points
(0.29 on average).

## What this shows

1. **At light pruning (≤ ~30% of parameters) ASF-S is competitive.** It matches or slightly beats HRank.
2. **At medium and heavy pruning (≥ 50%) ASF-S is worse than the baselines.** Sliming, SVD and HRank
   0.5–0.8 all dominate points on PG's front. At the paper's Table 1 operating point (~70% sparsity),
   Sliming reaches 37.0% against PG's 32.0%, at slightly *higher* compression.
3. This matches run 02's pre-fine-tune check, where PGI ≈ random and L1 > PGI. The gap is not an artefact
   of the fork's old baseline numbers.
4. **Headline claim:** no method, including the baselines, comes close to "baseline accuracy at 70%
   sparsity with FC-only fine-tuning" on CIFAR-100 Conv2Net. The best at ~70% sparsity is Sliming,
   15.5 points below the 52.52% baseline.
