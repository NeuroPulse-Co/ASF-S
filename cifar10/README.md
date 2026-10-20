# ASF-S on CIFAR-10: re-implementation in paper-checked steps

This directory is a from-scratch implementation of ASF-S (the seed paper) on the paper's own setting, CIFAR-10 with
Conv2Net. Each step implements one part of the method. Each step ends with a **gate**: its output is checked against
a number or figure from the paper before the next step starts.

It is standalone: nothing here imports from `cifar100/` or `min_scripts/`. The original CIFAR-10 scripts in
`min_scripts/` serve only as the specification for the `legacy` / `literal` modes, which reproduce what the authors
most likely ran.

## Running on the server

```bash
./cifar10/setup.sh            # once: .venv with the torch wheel matching the driver (2080: torch 2.5.1+cu121)
./cifar10/run.sh test         # unit tests
./cifar10/run.sh 0            # step 0, paper protocol → outputs/run01/conv2net_paper/step0_check_arch/gate.json
./cifar10/run.sh 1 --protocol clean --run-id run01
```

`run.sh` tees each log to `outputs/<run-id>/logs/`. Run steps 1, 8 and 9 under `tmux`.

## Steps and gates

| Step | What | Gate (paper) | Status |
|---|---|---|---|
| 0 | Model, data pipeline, metrics | params 183K, FLOPs 40.89M (Table 4); split sizes | code ready |
| 1 | Train Conv2Net (Table 3 recipe) | test acc 72.69% ± 1.5 (Table 4) | |
| 2 | Output-layer activations, correct / incorrect | ≥ S samples per (class, split) | |
| 3 | AFC matrix W, Eq. 2–3 | Fig. 3: truck ≈ automobile ≠ dog | |
| 4 | PG1, Eq. 5–8 | unit tests; Fig. 4 | |
| 5 | Contrast + PCA, Eq. 9–11 | top-4 explained variance ≈ 90% (§C.3) | |
| 6 | PGI, Eq. 12–13 (`literal` / `aligned`) | Fig. 11; PGI vs L1 / random rank correlation | |
| 7 | Prune, 9×9 τ grid | prune == mask; operating points of Tables 1 and 5 exist | |
| 8 | Fine-tune: FC-only (b), full (a) | ΔAcc −7.74 at 70% (Table 1); −3.78 at 75.8% (Table 5) | |
| 9 | Baselines: random, L1, L2, top-k | Table 5 rows; Fig. 7 | |

## Data protocol

| | `paper` (reproduce) | `clean` (report) |
|---|---|---|
| Train | 50k | 45k (stratified split, seed 42) |
| Early stopping, checkpoint and τ choice | test set | 5k val |
| PG1 activations | 1000/class train subset | 5k val |
| Test | used throughout | used once, for the final numbers |

The original scripts used the test set as the validation set. They also built PG1 from training images, where the
model makes few, atypical mistakes. `paper` mode reproduces this so we can compare against the paper's numbers;
`clean` mode gives unbiased ones. The difference between the two is itself a result.

## Choices the paper leaves open

These are all in `config.py`, each marked `[assumed]` with where the value comes from:
- S = 50, σ = 0.1, α = 0.5 (from the original scripts)
- early-stopping patience 15
- fine-tuning with Adam 1e-4, patience 10, at most 200 epochs

## Data pipeline

The `gpu` backend keeps the whole dataset on the GPU and applies the paper's augmentation per sample, in batches:
flip, crop with padding 4, ColorJitter(0.2, 0.2, 0.2, 0.1), rotation 15°. `tests/test_step0.py` checks each op
against `torchvision.transforms.functional`.

The `torchvision` backend (`--backend torchvision`) is the original PIL pipeline. It is kept as the reference.
Step 0 reports seconds per epoch for both.
