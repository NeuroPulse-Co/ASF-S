"""
paper_targets.py – the seed paper's numbers that each step is gated against.

Accuracies are in %. "Acc drop" is baseline minus pruned (positive = worse), as in
Table 1; Table 5 reports the same quantity as a negative ΔAcc.
"""

# Table 4: dense models on their test sets
DENSE = {
    "conv2net": dict(acc=72.69, params=183_000, flops=40.89e6),
    "conv6net": dict(acc=88.10, params=815_000, flops=79.47e6),
}

# §C.3: top-4 PCA components of the PG1 contrast matrix explain ~90% of variance
PCA_TOP4_EXPLAINED = 0.90

# Table 1: ASF-S at 70% sparsity
TABLE1_ASFS = {
    "conv2net": dict(acc_drop=7.74, sparsity=70.0, flops_reduction=57.01),
    "conv6net": dict(acc_drop=0.00, sparsity=70.0, flops_reduction=36.01),
}

# Table 5: top-FLOPs-reduction variant per method (ΔAcc, params↓ %, FLOPs↓ %)
TABLE5 = {
    "conv2net": {
        "asfs":   dict(d_acc=-3.78,  params_red=75.8, flops_red=82.7),
        "topk":   dict(d_acc=-16.12, params_red=91.5, flops_red=98.5),
        "random": dict(d_acc=-10.64, params_red=87.0, flops_red=96.3),
        "l1":     dict(d_acc=-12.99, params_red=90.0, flops_red=97.8),
        "l2":     dict(d_acc=-13.74, params_red=90.0, flops_red=97.8),
    },
    "conv6net": {
        "asfs":   dict(d_acc=-3.97,  params_red=67.0, flops_red=75.5),
        "topk":   dict(d_acc=-25.11, params_red=93.3, flops_red=98.7),
        "random": dict(d_acc=-28.55, params_red=95.3, flops_red=99.0),
        "l1":     dict(d_acc=-24.26, params_red=92.7, flops_red=98.2),
        "l2":     dict(d_acc=-21.05, params_red=92.7, flops_red=98.2),
    },
}
