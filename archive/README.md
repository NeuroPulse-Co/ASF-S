# Archive

These files are no longer in use. We keep them because the logs refer to them.

| File | What it was | Why it is archived |
|---|---|---|
| `kaggle/kaggle_conv2net.ipynb` | A notebook for running the Conv2Net pipeline on Kaggle GPUs | We moved to the department GPU server on 2026-09-15. A Kaggle run had hit the 12-hour session limit and lost all of its output, and the full threshold grid needs about 40–45 GPU-hours (see the 2026-09-15 entry in [the CIFAR-100 log](../studies/cifar100-port-audit/LOG.md)). |
| `README-cifar100-head-only-pipeline.md` | The old top-level README of this repository. It describes an early CIFAR-100 "head-only" pipeline under `CNN/cifar100_original_pipeline/`. | That folder layout no longer exists. The CIFAR-100 work now lives in [`studies/cifar100-port-audit`](../studies/cifar100-port-audit). |
