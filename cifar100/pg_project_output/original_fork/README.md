# Original fork results — do not edit

These are the CIFAR-100 result CSVs exactly as they were when we forked the repo.
They are the baseline we are trying to reproduce. **Treat this folder as read-only.**
New runs go in `../runs/`, never here.

| | |
|---|---|
| Fork point | commit `6e8b82a` (Charuka Herath, 2026-09-04) |
| CSVs added in | commit `c485447` "add baseline comparisons" (llckbhk, 2026-05-05) |
| Previously at | `pg_project_output/results/`, then `reference_results/` (renamed in `af77805`), now here |
| Seed | 42 in every pruning row |
| Hardware / env | not recorded |

## What is in here

### Training curves (unpruned baselines)

| File | Epochs | Best `val_acc` | Notes |
|---|---|---|---|
| `conv2net_metrics.csv` | 160 | **51.93%** @ epoch 156 | our run 01 reproduced this (52.52%) |
| `conv6net_metrics.csv` | 200 | **62.99%** @ epoch 197 | hit the 200-epoch cap |
| `vgg16_metrics.csv` | 108 | **58.17%** @ epoch 88 (final 55.99%) | LR schedule never stepped (finding #8) |
| `vgg16_metrics.csv.before_estimated_200.bak` | 81 | — | an earlier, different VGG16 run; origin unknown |

### Pruning results (post FC-only fine-tune)

| File | Setting | Param red. % | `val_acc` |
|---|---|---|---|
| `pg_conv2net_cifar100.csv` | 9 rows, conv1 thr 0.1, conv2 thr 0.1→0.9 | 10.4 → 98.5 | 50.03 → 8.40 |
| `pg_conv6net_cifar100.csv` | thr 0.7 | 51.9 | 44.40 |
| `pg_vgg16_cifar100.csv` | thr 0.7 | 99.5 | 1.71 (chance ≈ 1) |
| `hrank_conv2net_cifar100.csv` | ratio 0.2 | 21.3 | 47.58 |
| `svd_*` | ratio 0.7 | conv2 70.7 / conv6 78.3 / vgg 99.1 | 35.92 / 19.51 / 7.62 |
| `sliming_*` | ratio 0.7 | 70.7 / 78.3 / 99.1 | 36.16 / 24.99 / 9.49 |
| `snows_*` | ratio 0.7 | 70.7 / 78.3 / 99.1 | 33.69 / 20.93 / 10.92 |

## Known problems with these numbers

Details and evidence are in [`docs/engineering-log.md`](../../../docs/engineering-log.md) and
[`docs/cifar100-status.md`](../../../docs/cifar100-status.md).

- **`pre_acc` means nothing** (≈1% everywhere). `rebuild_fc` copies FC weights into the wrong
  columns (finding #1). Only `val_acc`, measured after the FC is retrained, is interpretable.
- **Param reduction % counts only the FC layer**, because the convs are frozen before `ptflops`
  runs (finding #2). This makes little difference for Conv2Net and a large one for VGG16.
- **PG rows use a threshold; the baselines use a ratio** (finding #5). PG thr 0.7 ≠ 70% sparsity.
- **PGI scores were built from S = 1 sample per class**, so the PG rows rest on degenerate PG1
  vectors (finding #3).
- **`svd_conv2net` and `svd_conv6net` have a broken header.** The first row's `svd,` is glued onto
  the end of the header (finding #9). To read them, strip the trailing `svd,` from the header and
  prepend `svd,` to the data row. The values in the table above were read that way.
- **`pg_conv2net` rows 3 and 7 are both labelled (0.1, 0.7)** but keep 43 and 2 conv2 filters.
  Row 3 sits between the 0.2 and 0.4 rows, so it is most likely thr 0.3 with the wrong label.
  `THRESHOLDS` was edited between runs.
