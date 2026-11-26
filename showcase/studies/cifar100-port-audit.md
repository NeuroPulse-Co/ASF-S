---
id: cifar100-port-audit
repo: ASF-S
title: Does ASF-S reproduce on CIFAR-100 once the ported code is audited and fixed?
workstream: W1
status: closed-negative
owner:
- Dewmina Wijekoon
started: '2026-09-13'
closed: '2026-09-29'
summary: We ported the Artificial Structure Function Search (ASF-S) authors' code to CIFAR-100, audited
  it against the paper and fixed three bugs. With the fixes, the method's filter scores were barely better
  than random, and simple baselines beat it at 50% or more of parameters removed.
question: On CIFAR-100 with Conv2Net, does ASF-S pruning with Principal Gradient Importance (PGI) and
  fine-tuning of only the output fully connected (FC) layer keep close to the unpruned accuracy at about
  70% parameter reduction, and does PGI choose better filters than random, L1-norm and the published baselines?
hypothesis: The paper reports that retraining only the output layer keeps full baseline accuracy at up
  to 70% sparsity in a convolutional network, because PGI keeps the filters that shape the output layer's
  Artificial Functional Connectivity (AFC). If that mechanism is general, it should carry over from CIFAR-10
  to CIFAR-100 once the code computes what the paper describes.
success_criterion: On the fixed code, ASF-S at about 70% parameter reduction stays within about 2 points
  of the unpruned Conv2Net test accuracy (roughly three times the 0.73-point noise we measured between
  repeated settings), PGI beats random and L1-norm filter choice at the same kept counts, and ASF-S is
  not dominated by HRank, SVD, network slimming or SNOWS at matched compression.
criterion: post_hoc
verdict: ASF-S does not reproduce on CIFAR-100 Conv2Net. After we fixed the classifier rebuild, the parameter
  counts and the single-sample PG1, ASF-S at 68.7% parameter reduction reached 40.46% test accuracy against
  52.52% unpruned (one seed), PGI chose filters only slightly better than random and worse than L1-norm,
  and network slimming, SVD and HRank beat it at 50% reduction or more. The most likely cause is the paper's
  PGI projection (Eq. 12), which is dimensionally undefined and which we deliberately left as the authors
  implemented it.
key_results:
- label: ASF-S on Conv2Net with 68.7% of parameters removed and only the FC layer fine-tuned (run 02,
    fixed code)
  value: 40.46
  unit: test accuracy (%)
  split: test
  n_seeds: 1
  compare:
    label: unpruned Conv2Net, our run 01
    value: 52.52
- label: Unpruned Conv2Net trained with the authors' CIFAR-100 script (run 01)
  value: 52.52
  unit: test accuracy (%)
  split: test
  n_seeds: 1
  compare:
    label: authors' fork, conv2net_metrics.csv
    value: 51.93
- label: ASF-S keeping 10 of 32 conv1 and 20 of 64 conv2 filters (69.0% of parameters removed), FC-only
    fine-tuning (run 02)
  value: 32.0
  unit: test accuracy (%)
  split: test
  n_seeds: 1
  compare:
    label: network slimming at 70.5% removed on the same fixed code (run 03)
    value: 37.02
- label: ASF-S filter choice keeping 30/57 filters, before any fine-tuning (run 02 check)
  value: 33.67
  unit: test accuracy (%)
  split: test
  n_seeds: 1
  compare:
    label: keeping the highest-L1-norm filters at the same counts
    value: 42.1
evidence:
- type: diagram
  title: Where the bugs sat in the authors' code path
  caption: 'Read the seven steps from top to bottom, in the order the ported scripts run Artificial Structure
    Function Search (ASF-S) on CIFAR-100. The amber steps are where our audit found problems. We fixed
    three before run 02: the first principal gradient (PG1) built from one image, the classifier rebuild,
    and the parameter count. We left the dashed one unchanged: the test set decides when fine-tuning stops,
    so every accuracy we report is the best test accuracy seen during fine-tuning, not a clean held-out
    estimate.'
  src: ../figures/cifar100-port-audit/evidence-01-where-the-bugs-sat-in-the-authors-code-path.svg
  source_url: https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/cifar100-port-audit/diagrams/method.excalidraw.svg
- type: table
  title: The other pruning methods, rerun on our fixed code
  caption: Four pruning methods from the authors' code, one seed each. At about 70% of parameters removed,
    network slimming reaches 37.02%, SVD 33.99% and SNOWS 32.34%. ASF-S at a similar size (10 and 20 filters
    kept, 69.0% removed) reaches 32.00%, so it does not beat them. Each accuracy is the best test accuracy
    seen during fine-tuning, because the code uses the test set to decide when to stop; this flatters
    every method equally.
  split: test
  columns:
  - key: method
    label: Method
  - key: kept_conv1
    label: Filters kept, layer 1 (of 32)
  - key: kept_conv2
    label: Filters kept, layer 2 (of 64)
  - key: param_red_pct
    label: Parameters removed (%)
  - key: val_acc
    label: Accuracy (%)
  labels:
    sliming: Network slimming
    svd: SVD
    snows: SNOWS
    hrank: HRank
  rows:
  - method: sliming
    kept_conv1: 10
    kept_conv2: 19
    param_red_pct: 70.5342
    val_acc: 37.02
  - method: svd
    kept_conv1: 10
    kept_conv2: 19
    param_red_pct: 70.5342
    val_acc: 33.99
  - method: snows
    kept_conv1: 10
    kept_conv2: 19
    param_red_pct: 70.5342
    val_acc: 32.34
  - method: hrank
    kept_conv1: 25
    kept_conv2: 51
    param_red_pct: 20.506
    val_acc: 46.88
  - method: hrank
    kept_conv1: 22
    kept_conv2: 44
    param_red_pct: 31.4869
    val_acc: 45.63
  - method: hrank
    kept_conv1: 16
    kept_conv2: 32
    param_red_pct: 50.2749
    val_acc: 42.23
  - method: hrank
    kept_conv1: 12
    kept_conv2: 25
    param_red_pct: 61.2061
    val_acc: 38.55
  - method: hrank
    kept_conv1: 6
    kept_conv2: 12
    param_red_pct: 81.4145
    val_acc: 29.21
  - method: hrank
    kept_conv1: 3
    kept_conv2: 6
    param_red_pct: 90.714
    val_acc: 19.68
  data: ../data/cifar100-port-audit/evidence-02-the-other-pruning-methods-rerun-on-our-fixed-cod.csv
  source_url: https://github.com/NeuroPulse-Co/ASF-S/tree/main/studies/cifar100-port-audit/cifar100/pg_project_output/runs/run03_2026-09-29_baselines/results
- type: table
  title: Every ASF-S setting we tried, after fixing the code
  caption: ASF-S at 81 combinations of pruning thresholds, one seed, with only the final layer fine-tuned.
    The best result with about two thirds of the parameters removed is 40.46% (30 and 20 filters kept,
    68.7% removed), against 52.52% for the unpruned network. As before, each accuracy is the best test
    accuracy seen during fine-tuning.
  split: test
  columns:
  - key: kept_conv1
    label: Filters kept, layer 1 (of 32)
  - key: kept_conv2
    label: Filters kept, layer 2 (of 64)
  - key: total_param_red_pct
    label: Parameters removed (%)
  - key: pre_acc
    label: Accuracy before fine-tuning (%)
  - key: val_acc
    label: Accuracy after fine-tuning (%)
  rows:
  - kept_conv1: 10
    kept_conv2: 3
    total_param_red_pct: 95.3272
    pre_acc: 1.01
    val_acc: 13.97
  - kept_conv1: 10
    kept_conv2: 47
    total_param_red_pct: 27.1466
    pre_acc: 1.93
    val_acc: 40.67
  - kept_conv1: 10
    kept_conv2: 31
    total_param_red_pct: 51.9395
    pre_acc: 1.1
    val_acc: 35.91
  - kept_conv1: 10
    kept_conv2: 20
    total_param_red_pct: 68.9847
    pre_acc: 1.26
    val_acc: 32.0
  - kept_conv1: 10
    kept_conv2: 9
    total_param_red_pct: 86.0298
    pre_acc: 1.5
    val_acc: 23.64
  - kept_conv1: 10
    kept_conv2: 5
    total_param_red_pct: 92.2281
    pre_acc: 1.1
    val_acc: 18.39
  - kept_conv1: 10
    kept_conv2: 57
    total_param_red_pct: 11.651
    pre_acc: 2.49
    val_acc: 41.57
  - kept_conv1: 10
    kept_conv2: 2
    total_param_red_pct: 96.8768
    pre_acc: 1.14
    val_acc: 10.82
  - kept_conv1: 10
    kept_conv2: 2
    total_param_red_pct: 96.8768
    pre_acc: 1.14
    val_acc: 10.79
  - kept_conv1: 26
    kept_conv2: 3
    total_param_red_pct: 95.2722
    pre_acc: 1.53
    val_acc: 13.86
  - kept_conv1: 26
    kept_conv2: 47
    total_param_red_pct: 26.7094
    pre_acc: 9.77
    val_acc: 47.3
  - kept_conv1: 26
    kept_conv2: 31
    total_param_red_pct: 51.6414
    pre_acc: 5.77
    val_acc: 43.86
  - kept_conv1: 26
    kept_conv2: 20
    total_param_red_pct: 68.782
    pre_acc: 2.6
    val_acc: 38.3
  - kept_conv1: 26
    kept_conv2: 9
    total_param_red_pct: 85.9227
    pre_acc: 1.6
    val_acc: 29.33
  - kept_conv1: 26
    kept_conv2: 5
    total_param_red_pct: 92.1557
    pre_acc: 1.23
    val_acc: 20.16
  - kept_conv1: 26
    kept_conv2: 57
    total_param_red_pct: 11.127
    pre_acc: 20.2
    val_acc: 48.84
  - kept_conv1: 26
    kept_conv2: 2
    total_param_red_pct: 96.8304
    pre_acc: 1.57
    val_acc: 10.21
  - kept_conv1: 26
    kept_conv2: 2
    total_param_red_pct: 96.8304
    pre_acc: 1.57
    val_acc: 10.36
  - kept_conv1: 24
    kept_conv2: 3
    total_param_red_pct: 95.2791
    pre_acc: 1.38
    val_acc: 12.72
  - kept_conv1: 24
    kept_conv2: 47
    total_param_red_pct: 26.7641
    pre_acc: 7.08
    val_acc: 46.66
  - kept_conv1: 24
    kept_conv2: 31
    total_param_red_pct: 51.6786
    pre_acc: 3.49
    val_acc: 43.06
  - kept_conv1: 24
    kept_conv2: 20
    total_param_red_pct: 68.8074
    pre_acc: 1.77
    val_acc: 37.53
  - kept_conv1: 24
    kept_conv2: 9
    total_param_red_pct: 85.9361
    pre_acc: 1.52
    val_acc: 28.58
  - kept_conv1: 24
    kept_conv2: 5
    total_param_red_pct: 92.1648
    pre_acc: 1.25
    val_acc: 19.51
  - kept_conv1: 24
    kept_conv2: 57
    total_param_red_pct: 11.1925
    pre_acc: 12.56
    val_acc: 47.83
  - kept_conv1: 24
    kept_conv2: 2
    total_param_red_pct: 96.8362
    pre_acc: 1.53
    val_acc: 9.96
  - kept_conv1: 24
    kept_conv2: 2
    total_param_red_pct: 96.8362
    pre_acc: 1.53
    val_acc: 9.79
  - kept_conv1: 22
    kept_conv2: 3
    total_param_red_pct: 95.2859
    pre_acc: 1.01
    val_acc: 12.09
  - kept_conv1: 22
    kept_conv2: 47
    total_param_red_pct: 26.8187
    pre_acc: 4.16
    val_acc: 44.82
  - kept_conv1: 22
    kept_conv2: 31
    total_param_red_pct: 51.7159
    pre_acc: 1.97
    val_acc: 40.97
  - kept_conv1: 22
    kept_conv2: 20
    total_param_red_pct: 68.8327
    pre_acc: 1.69
    val_acc: 35.56
  - kept_conv1: 22
    kept_conv2: 9
    total_param_red_pct: 85.9495
    pre_acc: 1.25
    val_acc: 26.4
  - kept_conv1: 22
    kept_conv2: 5
    total_param_red_pct: 92.1738
    pre_acc: 1.0
    val_acc: 19.33
  - kept_conv1: 22
    kept_conv2: 57
    total_param_red_pct: 11.258
    pre_acc: 4.29
    val_acc: 47.28
  - kept_conv1: 22
    kept_conv2: 2
    total_param_red_pct: 96.842
    pre_acc: 1.21
    val_acc: 9.97
  - kept_conv1: 22
    kept_conv2: 2
    total_param_red_pct: 96.842
    pre_acc: 1.21
    val_acc: 10.11
  - kept_conv1: 17
    kept_conv2: 3
    total_param_red_pct: 95.3031
    pre_acc: 1.19
    val_acc: 13.42
  - kept_conv1: 17
    kept_conv2: 47
    total_param_red_pct: 26.9553
    pre_acc: 2.92
    val_acc: 40.99
  - kept_conv1: 17
    kept_conv2: 31
    total_param_red_pct: 51.8091
    pre_acc: 1.67
    val_acc: 38.79
  - kept_conv1: 17
    kept_conv2: 20
    total_param_red_pct: 68.896
    pre_acc: 1.22
    val_acc: 34.14
  - kept_conv1: 17
    kept_conv2: 9
    total_param_red_pct: 85.983
    pre_acc: 1.23
    val_acc: 25.98
  - kept_conv1: 17
    kept_conv2: 5
    total_param_red_pct: 92.1964
    pre_acc: 1.25
    val_acc: 18.02
  - kept_conv1: 17
    kept_conv2: 57
    total_param_red_pct: 11.4217
    pre_acc: 3.65
    val_acc: 43.58
  - kept_conv1: 17
    kept_conv2: 2
    total_param_red_pct: 96.8565
    pre_acc: 1.53
    val_acc: 11.46
  - kept_conv1: 17
    kept_conv2: 2
    total_param_red_pct: 96.8565
    pre_acc: 1.53
    val_acc: 10.73
  - kept_conv1: 12
    kept_conv2: 3
    total_param_red_pct: 95.3203
    pre_acc: 1.07
    val_acc: 13.3
  - kept_conv1: 12
    kept_conv2: 47
    total_param_red_pct: 27.0919
    pre_acc: 2.27
    val_acc: 40.54
  - kept_conv1: 12
    kept_conv2: 31
    total_param_red_pct: 51.9023
    pre_acc: 1.32
    val_acc: 36.82
  - kept_conv1: 12
    kept_conv2: 20
    total_param_red_pct: 68.9594
    pre_acc: 1.18
    val_acc: 31.93
  - kept_conv1: 12
    kept_conv2: 9
    total_param_red_pct: 86.0165
    pre_acc: 1.42
    val_acc: 23.86
  - kept_conv1: 12
    kept_conv2: 5
    total_param_red_pct: 92.219
    pre_acc: 1.25
    val_acc: 18.45
  - kept_conv1: 12
    kept_conv2: 57
    total_param_red_pct: 11.5855
    pre_acc: 2.26
    val_acc: 41.72
  - kept_conv1: 12
    kept_conv2: 2
    total_param_red_pct: 96.871
    pre_acc: 1.43
    val_acc: 11.02
  - kept_conv1: 12
    kept_conv2: 2
    total_param_red_pct: 96.871
    pre_acc: 1.43
    val_acc: 11.36
  - kept_conv1: 30
    kept_conv2: 3
    total_param_red_pct: 95.2584
    pre_acc: 1.85
    val_acc: 16.23
  - kept_conv1: 30
    kept_conv2: 47
    total_param_red_pct: 26.6002
    pre_acc: 21.38
    val_acc: 49.18
  - kept_conv1: 30
    kept_conv2: 31
    total_param_red_pct: 51.5668
    pre_acc: 13.64
    val_acc: 45.75
  - kept_conv1: 30
    kept_conv2: 20
    total_param_red_pct: 68.7314
    pre_acc: 5.9
    val_acc: 40.46
  - kept_conv1: 30
    kept_conv2: 9
    total_param_red_pct: 85.8959
    pre_acc: 2.79
    val_acc: 29.59
  - kept_conv1: 30
    kept_conv2: 5
    total_param_red_pct: 92.1376
    pre_acc: 1.4
    val_acc: 22.19
  - kept_conv1: 30
    kept_conv2: 57
    total_param_red_pct: 10.996
    pre_acc: 33.67
    val_acc: 51.12
  - kept_conv1: 30
    kept_conv2: 2
    total_param_red_pct: 96.8189
    pre_acc: 1.94
    val_acc: 12.96
  - kept_conv1: 30
    kept_conv2: 2
    total_param_red_pct: 96.8189
    pre_acc: 1.94
    val_acc: 13.01
  - kept_conv1: 6
    kept_conv2: 3
    total_param_red_pct: 95.3409
    pre_acc: 1.02
    val_acc: 15.25
  - kept_conv1: 6
    kept_conv2: 47
    total_param_red_pct: 27.2559
    pre_acc: 1.61
    val_acc: 38.76
  - kept_conv1: 6
    kept_conv2: 31
    total_param_red_pct: 52.0141
    pre_acc: 1.12
    val_acc: 36.57
  - kept_conv1: 6
    kept_conv2: 20
    total_param_red_pct: 69.0353
    pre_acc: 0.93
    val_acc: 30.24
  - kept_conv1: 6
    kept_conv2: 9
    total_param_red_pct: 86.0566
    pre_acc: 1.29
    val_acc: 23.93
  - kept_conv1: 6
    kept_conv2: 5
    total_param_red_pct: 92.2462
    pre_acc: 1.21
    val_acc: 17.97
  - kept_conv1: 6
    kept_conv2: 57
    total_param_red_pct: 11.782
    pre_acc: 1.41
    val_acc: 40.37
  - kept_conv1: 6
    kept_conv2: 2
    total_param_red_pct: 96.8883
    pre_acc: 1.05
    val_acc: 10.97
  - kept_conv1: 6
    kept_conv2: 2
    total_param_red_pct: 96.8883
    pre_acc: 1.05
    val_acc: 10.59
  - kept_conv1: 4
    kept_conv2: 3
    total_param_red_pct: 95.3478
    pre_acc: 1.01
    val_acc: 16.05
  - kept_conv1: 4
    kept_conv2: 47
    total_param_red_pct: 27.3105
    pre_acc: 1.22
    val_acc: 37.79
  - kept_conv1: 4
    kept_conv2: 31
    total_param_red_pct: 52.0513
    pre_acc: 1.13
    val_acc: 35.85
  - kept_conv1: 4
    kept_conv2: 20
    total_param_red_pct: 69.0607
    pre_acc: 1.2
    val_acc: 28.97
  - kept_conv1: 4
    kept_conv2: 9
    total_param_red_pct: 86.07
    pre_acc: 1.21
    val_acc: 22.25
  - kept_conv1: 4
    kept_conv2: 5
    total_param_red_pct: 92.2552
    pre_acc: 1.04
    val_acc: 17.94
  - kept_conv1: 4
    kept_conv2: 57
    total_param_red_pct: 11.8475
    pre_acc: 1.17
    val_acc: 38.68
  - kept_conv1: 4
    kept_conv2: 2
    total_param_red_pct: 96.8941
    pre_acc: 0.99
    val_acc: 13.17
  - kept_conv1: 4
    kept_conv2: 2
    total_param_red_pct: 96.8941
    pre_acc: 0.99
    val_acc: 12.58
  data: ../data/cifar100-port-audit/evidence-03-every-asf-s-setting-we-tried-after-fixing-the-co.csv
  source_url: https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/cifar100-port-audit/cifar100/pg_project_output/runs/run02_2026-09-28_rtx2080_S50_fixes/results/pg_conv2net_cifar100.csv
runs:
- id: 20260916-0031-intellisense08-EWISPro9900G-run01-conv2net
  started: '2026-09-16 00:31:00+05:30'
  host: intellisense08-EWISPro9900G
  seed: null
  commit: unknown
  exit_code: null
  backfilled: true
- id: 20260928-1314-intellisense08-EWISPro9900G-run02-conv2net
  started: '2026-09-28 13:14:00+05:30'
  host: intellisense08-EWISPro9900G
  seed: 42
  commit: 1d43cb7
  exit_code: null
  backfilled: true
- id: 20260929-1207-intellisense08-EWISPro9900G-run03-baselines
  started: '2026-09-29 12:07:00+05:30'
  host: intellisense08-EWISPro9900G
  seed: 42
  commit: 767f8af
  exit_code: null
  backfilled: true
code_url: https://github.com/NeuroPulse-Co/ASF-S/tree/main/studies/cifar100-port-audit
---

<!-- Generated by record.py promote from studies/cifar100-port-audit/study.yaml and FINDINGS.md. Edit those files, not this one. -->

## Verdict

We asked whether Artificial Structure Function Search (ASF-S), the structured pruning method we build on, stays close to the unpruned accuracy on CIFAR-100. The test was to prune with its Principal Gradient Importance (PGI) score and then retrain only the output fully connected (FC) layer. We ported the authors' code to CIFAR-100 and read it line by line against the paper. We then fixed three bugs that made the reported numbers mean something other than their column names.

**With those fixes, ASF-S does not reproduce on CIFAR-100 Conv2Net.**
- With 68.7% of parameters removed, it reaches 40.46% test accuracy. That is 12.1 points below our unpruned 52.52% (one seed).
- Its filter choice is only slightly better than random, and worse than plain L1-norm.
- Network slimming, SVD and HRank all beat it once half or more of the parameters are removed.

The hypothesis did not hold, so we closed the study as negative.

Two points of fairness matter here:
- **This is not the paper's own setting.** The paper reports its convolutional results on CIFAR-10 and MNIST, not CIFAR-100. This study therefore tests whether the method carries over to a harder dataset; it does not re-run the paper's own table.
- **We started from a fork.** The CIFAR-100 code we started from is a fork of the authors' repository. Several of the problems below are porting issues or gaps in the paper's description, not errors in the paper's reported results.

## Evidence

**What we ran.** All runs used Conv2Net (two convolutional layers and one FC layer) on CIFAR-100, on one RTX 2080 server, with seed 42 in every pruning row. We kept the authors' fork results unchanged as the reference, in [`original_fork/`](https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/cifar100-port-audit/cifar100/pg_project_output/original_fork/README.md).

- [Run 01](https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/cifar100-port-audit/runs/20260916-0031-intellisense08-EWISPro9900G-run01-conv2net/run.yaml) trained Conv2Net with the unmodified code and computed the first principal gradient (PG1) vectors. These showed the single-sample collapse described below.
- [Run 02](https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/cifar100-port-audit/runs/20260928-1314-intellisense08-EWISPro9900G-run02-conv2net/run.yaml) applied fixes 1 to 3 and ran the full 81-point ASF-S threshold grid, reusing run 01's checkpoint.
- [Run 03](https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/cifar100-port-audit/runs/20260929-1207-intellisense08-EWISPro9900G-run03-baselines/run.yaml) re-ran the baselines (HRank, SVD, network slimming and SNOWS) on the same checkpoint and the same fixed code.

**The code audit.** We found fifteen issues in total. The table below lists the ones that change what a number means. The full list, with file and line references, is in [`LOG.md`](https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/cifar100-port-audit/LOG.md) and [`docs/cifar100-status.md`](https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/cifar100-port-audit/docs/cifar100-status.md#other-findings).

| # | Problem | Effect on results | What we did |
|---|---|---|---|
| 1 | `rebuild_fc` copied the *first* FC columns instead of the columns belonging to the kept channels | After pruning, the FC layer was rewired at random. Every accuracy measured before fine-tuning sat near chance (0.66% to 1.83% in the fork) | Fixed: we copy the kept channels' columns. The pruned network now matches the unpruned one with those channels silenced (largest logit difference about 1e-7) |
| 2 | The pruned model's parameters were counted after the convolutions were frozen, so only the FC layer was counted | Sparsity was overstated: by under 1.2% on Conv2Net, but severely on VGG16 | Fixed: we count every parameter in both models |
| 3 | PG1 was built from one image per class and split (S = 1), although the paper's Eq. 1 needs S samples | Every PG1 vector had one or two distinct values, and 26 of 100 "incorrect" vectors were constant | Fixed: we use up to 50 samples |
| 4 | Eq. 12 multiplies three vectors indexed by unrelated things (class, FC input feature and filter weight), and the code cuts them to a common prefix | Each filter's score becomes, in effect, a fixed random projection of its weights | Left unchanged on purpose. Any repair changes the method, so it is a question for the authors |
| 5 | ASF-S prunes by a threshold on the normalised score, while the baselines prune to an exact ratio | A threshold of 0.7 is not 70% sparsity | We compared methods at achieved sparsity, using Pareto dominance |
| 6 | The "validation" loader is the CIFAR-100 test set, and it is used for early stopping | Every reported accuracy is the best test accuracy seen during fine-tuning, which is optimistic | Left unchanged, and noted on every number here |

**PG1 after fix 3.** Run 02 confirmed that the fix worked:
- No "incorrect" vector was constant, where 26 of 100 had been before.
- The first principal component of the correct-minus-incorrect contrast explained 38.8% of the variance, up from 5.0%.
- The graph was well connected at the code's kernel width: 87% to 88% of off-diagonal affinities were above 0.01. So the concern that the graph might be nearly empty did not come true.

**Accuracy after pruning.** The next table sets our fixed-code results beside the authors' fork results at similar settings. It shows how far ASF-S falls short of baseline accuracy, and that our fixes did not move the final numbers much. All values are test accuracy in percent, from one seed, after FC-only fine-tuning unless stated.

| Setting | Ours | Authors' fork |
|---|---|---|
| Unpruned Conv2Net | 52.52 (run 01) | 51.93 |
| ASF-S, lightest pruning | 51.12 at 11.0% of parameters removed (run 02) | 50.03 at 10.4% removed |
| ASF-S, about two thirds removed | 40.46 at 68.7% removed (run 02) | 43.31 at 66.0% removed |
| ASF-S, 10/20 filters kept (69.0% removed) | 32.00 (run 02) | not run |
| Network slimming, ratio 0.7 (70.5% removed) | 37.02 (run 03) | 36.16 |
| SVD, ratio 0.7 | 33.99 (run 03) | 35.92 |
| SNOWS, ratio 0.7 | 32.34 (run 03) | 33.69 |
| HRank, ratio 0.2 | 46.88 (run 03) | 47.58 |

Our baselines land within −1.9 to +0.9 points of the fork's. This tells us that once the FC layer is retrained, the `rebuild_fc` bug hardly changes the final accuracy. Repeated grid settings that kept identical filter counts differed by 0.29 points on average and by 0.73 at most. So we treat differences under about one point as noise.

**Does PGI choose better filters?** To separate the filter choice from the FC retraining, we evaluated the pruned networks *before* any fine-tuning. Each network kept exactly ASF-S's filter counts, and we chose the filters by PGI, by highest L1-norm, or at random (10 draws).
- Over 8 settings, PGI averaged only 1.8 points above the random mean, and 30 of the 80 random draws matched or beat it.
- L1-norm beat PGI in 7 of the 8 settings.
- At the lightest setting (30 of 32 and 57 of 64 filters kept), PGI reached 33.67%, against 42.10% for L1-norm and 28.66% ± 5.77 for random.

**Against the baselines at matched compression (run 03).**
- ASF-S is competitive only at light pruning. Its points beat HRank at ratios 0.2 and 0.3, though the first margin is within noise.
- From about 50% of parameters removed, HRank (ratios 0.5 to 0.8), SVD and network slimming each dominate points on ASF-S's Pareto front.
- At the paper's operating point of about 70%, network slimming reaches 37.02% against ASF-S's 32.00%, while removing slightly *more* parameters (70.5% against 69.0%).

This study has no figures, because the archived runs contain only CSV files and PG1 arrays.

## Why we think this happened

Our main explanation is Eq. 12, the step that turns PG1 into a score for each filter. The paper scores a filter as the inner product of its weights with a PCA component of PG1, multiplied element by element with the mean output-layer weight. For Conv2Net these three vectors have different lengths:
- 27 or 288: one filter's weights
- 100: one entry per class
- 16,384: one entry per FC input feature

Their entries refer to unrelated things. The paper does not say how they line up, and the code resolves the mismatch by taking the first entries of each. The resulting score carries almost none of the PG1 structure. That fits what we measured: once fix 3 made PG1 meaningful, the final accuracy did not change, and the filter choice stayed close to random.

We are fairly confident in this explanation, because the arithmetic is unambiguous. However, we have not tested a repaired Eq. 12, so we cannot yet say whether a version with consistent dimensions would work.

The rest of the shortfall has several sources:
- **Our porting choices or the fork's.** The code uses 8 PCA components instead of the paper's 4. It passes a row-normalised diffusion operator to an eigensolver that assumes a symmetric matrix. It draws the PG1 samples from the training set.
- **Gaps in the paper.** The paper does not state the sample count, and it does not define the dimensions in Eq. 12.
- **The setting.** With 100 output units, the contrast matrix has no small set of dominant directions. The paper's "four components explain about 90%" refers to CIFAR-10's 10 output units.

Separately, even the best baseline here loses 15.5 points at about 70% removal. On CIFAR-100 Conv2Net, recovering full accuracy at that sparsity by retraining only the FC layer looks out of reach for every method we tried, not only ASF-S.

What would change our mind is a version of Eq. 12, agreed with the authors, that beats L1-norm and random selection at the same kept counts over several seeds.

## Limits

- We used one model (Conv2Net), one dataset (CIFAR-100) and one seed per setting. We never ran the Conv6Net or VGG16 grids on the fixed code. The VGG16 training script also never steps its learning-rate schedule.
- Every accuracy is the best *test* accuracy seen during fine-tuning, because the code uses the test set for early stopping. This flatters all methods equally, but no number here is a clean held-out estimate.
- CIFAR-100 is not the paper's setting, so this study does not show that the paper's CIFAR-10 numbers are wrong.
- We compared PGI with random and L1-norm filter choices only before fine-tuning. The comparison after FC fine-tuning, at ASF-S's exact kept counts, is still open.
- Our network slimming and SNOWS baselines are the fork's approximations, not the published methods. The slimming score is singular-value energy times the batch-norm scale, and the SNOWS score is the mean squared activation.

## What this means for the project

This study closes the CIFAR-100 part of workstream W1. We decided not to spend GPU time on Conv6Net or VGG16 with the current Eq. 12, because on the model where we can check cheaply, the filter scores it produces are close to random. Two actions follow:

1. Before publishing this page, we will raise three questions with the authors: Eq. 12, the sample count S, and which script produced which figure.
2. We carry these lessons into workstream W2. There, a way of scoring filters by their effect on the output layer's functional connectivity, with consistent dimensions, is compared with simple scores under a careful protocol.

**Acknowledgements.** Dewmina Wijekoon did most of the porting, auditing and runs. The GitHub users `llckbhk` and `ravindusw` made smaller contributions to this code; `llckbhk` added the CIFAR-100 baseline scripts in the fork.

**Reference.** *Artificial Structure Function Search: Preserving Artificial Functional Connectivity for Structured Pruning* (2026).
