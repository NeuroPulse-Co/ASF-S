# afcprune

AFC-preserving structured pruning. It continues the ASF-S reproductions (`cifar10/`, `cifar100/`, `mnist/`):
it keeps the seed paper's idea (preserve the output layer's Artificial Functional Connectivity so the pruned
layers need no retraining) and tests it under a protocol built to survive review. The package is
self-contained (it imports nothing from the rest of this repo) and will move to its own repo.

Plan: `docs/` + LOG.md. Positioning against prior work: [`docs/related_work.md`](docs/related_work.md).

## Status

| Stage | What | Status |
|---|---|---|
| S0 | Skeleton, ports, protocol guard, golden regression | **done**, 73 tests |
| S1 | Dense models (3 clean seeds + run01 + weak) and the selector × tier × budget baseline factorial | next |
| S2 | Does AFC carry information beyond logit matching? (go / no-go) | |
| S3 | Closed-form head and recovery tiers | |
| S4 | Multi-layer AFC, FC-redundancy, iterative, allocation, class-conditional | |
| S5 | Frozen configs, one test-set look, Conv6Net, paper scorecard | |

## Layout

| Module | Contents |
|---|---|
| `afc.py` | Eq. 1–8 (cosine AFC, Gaussian W, diffusion K, PG1), plus the top-k embedding, principal-angle subspace similarity and bootstrap PG1 stability |
| `contrast.py` | Eq. 9–11: correct − incorrect contrast ΔG, PCA |
| `scores/` | registry `scorer(model, ctx) → {conv: (F,)}`: `pgi_aligned`, `pgi_literal`, `l1`, `l2`, `random`, `bn_gamma`, `fpgm`, `taylor`, `hrank`, `act_mean` |
| `surgery.py` | exact filter removal (conv, BN, next conv, FC columns); `masked()` is the full-size equivalent |
| `allocate.py` | per-layer kept counts for a total-params↓ budget, chosen on val by one criterion for all selectors |
| `recovery.py` | tiers T0–T4: `ridge_head`, `bn_recalibrate`, `ls_compensate`, `finetune(fc / fc_frozen / full)` |
| `protocol.py` | splits; the test set can only be read inside `allow_test(reason)`, and every access is logged |
| `models.py` `data.py` `train.py` `metrics.py` | ported from `cifar10/` (Conv2Net / Conv6Net, GPU augmentation, fit loop, params / FLOPs / ECE) |

## Recovery tiers (the headline is T1)

| Tier | What may change after pruning |
|---|---|
| T0 | nothing |
| T1 | only the output FC: `fc_frozen` gradient FT or closed-form `ridge_head` |
| T2 | T1 + BatchNorm statistics re-estimated (forward passes only) |
| T3 | T2 + closed-form least-squares refit of the next conv's weights (no gradients) |
| T4 | full fine-tuning |

run01's −10.50 (cifar10 step 8) used mode `fc`, in which BN statistics adapt during FC training. That sits
between T1 and T2.

## Setup and tests

```bash
VENV=$PWD/afcprune/.venv bash cifar10/setup.sh --cuda cpu   # laptop; on the server let it detect CUDA
cd afcprune && .venv/bin/python -m pytest -q
```

`tests/golden/golden.npz` freezes the original `cifar10/asfs` outputs on synthetic inputs. `test_golden.py`
checks that afcprune reproduces them exactly. Regenerate it only from this repo, with
`tests/golden/make_golden.py`.
