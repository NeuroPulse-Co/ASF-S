# ASF-S Reproduction

This repository is where we tried to reproduce **Artificial Structure Function Search (ASF-S)**, the structured pruning method our project builds on, and where we recorded what we learned when the results did not come out as the paper reports. ASF-S scores each convolutional filter with **Principal Gradient Importance (PGI)**, which measures how the filter shapes the *Artificial Functional Connectivity (AFC)* of the output layer. The paper claims that this keeps about 70% of parameters removable while retraining only the final layer.

Part of **NeuroPulse**, a final-year research project on structure-function-aware pruning for continual learning at the University of Moratuwa. Project site: https://neuropulse-co.github.io · Organisation: https://github.com/NeuroPulse-Co

**Workstream:** W1, Reproducing ASF-S. The follow-on work, a careful comparison of AFC-based filter selection against simple scores (W2), lives in its own repository, [**AFC-Prune**](https://github.com/NeuroPulse-Co/AFC-Prune). Its history up to 2026-11-25 is also in this repository's history under `afcprune/`.

## Studies

| Study | Question | Status | Owner |
|---|---|---|---|
| [`cifar10-repro`](studies/cifar10-repro/) | Does a step-by-step re-implementation reproduce the paper's Conv2Net CIFAR-10 result? | closed-negative: steps 0–7 match the paper; retraining only the last layer loses 10.50 points against the paper's 3.78 | Dewmina Wijekoon |
| [`cifar100-port-audit`](studies/cifar100-port-audit/) | Does the authors' code, ported to CIFAR-100, keep accuracy after pruning? | closed-negative: after fixing three bugs, 12.1 points are lost at 68.7% of parameters removed | Dewmina Wijekoon |
| [`mnist-lenet-sanity`](studies/mnist-lenet-sanity/) | Does the method work on the simplest network in the paper, LeNet-300-100 on MNIST? | closed-negative: the baseline reproduces exactly, but no ASF-S variant beats magnitude pruning | Ravindu Wanigasooriya |
| [`bert-pg-transfer`](studies/bert-pg-transfer/) | Does principal-gradient pruning transfer to BERT's feed-forward layers? | closed-inconclusive: every method loses under about one point, so they cannot be told apart | Charuka09 (record maintained by Dewmina Wijekoon) |

Each study folder has a `study.yaml` (question, hypothesis, verdict, headline numbers), a `FINDINGS.md` (the write-up), a `LOG.md` (the dated lab log) and `runs/` (one record per run). The plain-language versions are on the project site.

**A note on fairness.** A failed reproduction has several possible causes: our bugs, details the paper does not state, or errors in the paper. Each `FINDINGS.md` separates what we ruled out from what we only suspect. We will share these findings with the paper's authors before the write-ups are published on the project site.

## Layout

| Path | What it is |
|---|---|
| `studies/<id>/` | One folder per study: its code, records and small results |
| `reference/min_scripts/` | The authors' original CIFAR-10 scripts, shared with their permission. We treat them as the specification of what was actually computed. |
| `archive/` | Files no longer in use that the logs still refer to |
| `showcase/` | Generated pages that the project website publishes (do not edit by hand) |
| `tools/record.py` | Our record tool (see below) |
| `plan.md` | The project-wide plan |

## How to run

Each study has its own instructions in its README. For example:

```bash
./studies/cifar10-repro/setup.sh          # once: creates a venv with the right torch wheel
./studies/cifar10-repro/run.sh test       # unit tests
./studies/cifar10-repro/run.sh 1          # step 1, paper protocol
```

New runs should be wrapped in the record tool, so that the code version, seed and environment are captured automatically:

```bash
uv run tools/record.py run <study> --tag <tag> --seed 42 -- <command>
```

Large outputs (checkpoints, activation dumps) are never committed. They stay on the GPU server, and the run records say where.

## How we keep records

We follow the [NeuroPulse record standard](https://github.com/NeuroPulse-Co/.github/blob/main/standards/RECORDS.md) and [writing guide](https://github.com/NeuroPulse-Co/.github/blob/main/standards/STYLE.md). `uv run tools/record.py check` validates every record, and it also runs in CI on every push.

The paper: *Artificial Structure Function Search: Preserving Artificial Functional Connectivity for Structured Pruning* (2026).
