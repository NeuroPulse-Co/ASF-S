# Run 01 — Conv2Net / CIFAR-100, department server, unmodified code (S = 1)

Our first reproduction, run with the fork's code **unchanged**. It stopped after step 4, once
the PG1 data showed the S = 1 collapse. It is kept here as evidence. Nothing in it should be
used as a result.

| | |
|---|---|
| Dates | 2026-09-16 (step 1) → 2026-09-18 (steps 2, 4) |
| Code | `cifar_100` branch at `ee13aae` / `b4a42c0`, no pipeline fixes applied |
| Host | `intellisense08-EWISPro9900G`, NVIDIA RTX 2080 8 GB, driver 535.309.01 (CUDA 12.2) |
| Env | torch 2.5.1+cu121, Python 3.10, venv from `scripts/run_pipeline.sh --setup` |
| Model | Conv2Net only |

## Steps

| Step | Script | Status | Output here |
|---|---|---|---|
| 1 train | `train_cifar100.py` | ✅ 165 epochs, 64 min, best **52.52%** @ 152 (fork: 51.93%) | `results/conv2net_metrics.csv`, `trained_models/conv2net_best.pth` |
| 2 activations | `activation_ext_cifar100.py` | ✅ second attempt, 13 min. The first died at class 66 when `/home` filled up | not kept (~39 GB) |
| 3 affinity | `cosine_sim_ext_cifar100.py` | ⏭ skipped (the pruner does not read it) | — |
| 4 PG1 | `pg_ext_cifar100.py` | ✅ 200/200 ok in 2.7 s | `pg_data/pg1_summary.csv`, `pg_data/conv2net/{correct,incorrect}/pg1_data/*_fc_pg1.npy` |
| 5 HRank | `hrank_cifar100.py` | ⏭ not run | — |
| 6 PG pruning | `pg_pruning_cifar100.py` | ⏭ not run: blocked by the findings below | — |

`trained_models/` and the `.npy` files are gitignored, so they exist only on the machines they
were copied to. The server step logs (`logs/<run-id>/`) are still on the server. Copy them in here.

## What this run showed

- Training reproduces the fork (+0.59 pts, within ordinary cuDNN variance).
- **PG1 is degenerate because it is built from one sample (S = 1).** Every vector has 1–2 distinct
  values, 26/100 `incorrect` vectors are constant, and PCA(8) over ΔG explains only 30.9% of the
  variance. The full analysis is in [`docs/cifar100-status.md`](../../../../docs/cifar100-status.md#the-s-problem).

The checkpoint is valid; none of the known bugs touch training. A copy stays in the live
`trained_models/` so run 02 can start from step 2.
