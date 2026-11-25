# Reference code

## `min_scripts/`: the original ASF-S scripts

These four scripts are the CIFAR-10 implementation of Artificial Structure Function Search (ASF-S), as we received it from the authors of the ASF-S paper. The authors kindly allowed us to keep it here publicly. All credit for this code belongs to them.

We treat these scripts as **the specification** of what the authors actually computed. Our re-implementations reproduce them where the paper's text is ambiguous: the "legacy" or "literal" modes in [`studies/cifar10-repro`](../studies/cifar10-repro) and in AFC-Prune follow these scripts line by line. The golden tests check that those modes give the same numbers.

| Script | Pipeline step |
|---|---|
| `activation_ext3.py` | Extract output-layer activations for correctly and incorrectly classified samples |
| `cosine_sim_ext_2.py` | Build the Artificial Functional Connectivity (AFC) similarity matrices |
| `pg_ext_2.py` | Compute the first principal gradient (PG1) of each AFC graph |
| `pg_pruning_conv2.py` | Score filters with Principal Gradient Importance (PGI), then prune and fine-tune Conv2Net |

We do not modify these files. Bugs and ambiguities we found in them are documented in the studies that use them, not fixed here.
