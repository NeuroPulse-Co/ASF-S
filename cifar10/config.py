"""
config.py – every setting of the CIFAR-10 ASF-S re-implementation.

Values the paper states are marked [paper §...]. Values it leaves open are marked
[assumed] with where they come from, so a reader can tell a reproduction choice
from a paper fact. Paper numbers used as pass/fail gates live in paper_targets.py.
"""

from pathlib import Path

ROOT        = Path(__file__).resolve().parent
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

SEED        = 42                                   # [assumed] original scripts
NUM_CLASSES = 10
CLASSES     = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# ── Data ──────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)                    # [paper App. C]
STD  = (0.2023, 0.1994, 0.2010)                    # [paper App. C]

# Augmentation [paper App. C]: flip, crop 32 pad 4, ColorJitter(0.2,0.2,0.2,0.1), rotation 15°
JITTER   = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
ROTATION = 15.0
CROP_PAD = 4

# "gpu": whole dataset on the GPU, per-sample augmentation in batches (fast).
# "torchvision": the original PIL transforms + DataLoader (reference, ~5x slower).
AUGMENT_BACKEND = "gpu"
NUM_WORKERS     = 4                                # torchvision backend only

# Protocol (see README "Data protocol"):
#   paper – train on 50k, test set doubles as validation, PG1 from a train subset
#           (what the original scripts did; used to reproduce the paper's numbers)
#   clean – stratified 45k/5k train/val split, test set used once at the end
PROTOCOL  = "paper"
VAL_SIZE  = 5000                                   # clean protocol only

# ── Training [paper Table 3] ──────────────────────────────────
TRAIN = {
    "conv2net": dict(lr=2e-4, weight_decay=5e-4, batch_size=128, max_epochs=100),
    "conv6net": dict(lr=3e-4, weight_decay=5e-4, batch_size=128, max_epochs=100),
}
PLATEAU_PATIENCE = 10                              # [paper App. C]
PLATEAU_FACTOR   = 0.1                             # [paper App. C]
EARLY_STOP_PATIENCE = 15                           # [assumed] paper says "early stopping", no patience

# ── ASF-S ─────────────────────────────────────────────────────
PG_LAYER        = {"conv2net": "fc", "conv6net": "fc2"}   # the output layer [paper §4]
ACT_PER_CLASS   = 1000      # [assumed] original activation_ext3.py: 1000/class train subset
S_SAMPLES       = 50        # [assumed] S in Eq. 1; original pg_ext_2.py max_samples
# σ in Eq. 3 is not given in the paper. The original scripts' 0.1 disconnects the 10×10
# output-layer graph (λ1 = λ2 = λ3 = 1, PG1 undefined; LOG.md, step 4). "median" = median of
# the off-diagonal 1 − A over all (class, split) groups, computed in step 3 and saved there.
SIGMA           = "median"  # [assumed] chosen by diag_sigma: run01 → 0.831, stable PG1
SIGMA_LEGACY    = 0.1       # original scripts; kept for the legacy reproduction
ALPHA           = 0.5       # [assumed] Eq. 6; original scripts
EPS             = 1e-6      # [assumed] Eq. 3
PCA_K           = 4         # [paper §C.3] top 4 components, ~90% variance
THRESHOLDS      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # [paper §B.2]

PG_MODE      = "paper"      # paper: eigh(K), unit norm | legacy: row-normalised M, λ^t scaling
PGI_MODE     = "aligned"    # literal: original prefix truncation | aligned: see asfs/pgi.py
ABS_OF_MEAN  = False        # True reproduces the original |mean_k s| instead of Eq. 13's mean_k |s|

# ── Fine-tuning [assumed: original pg_pruning_conv2.py] ───────
FINETUNE = dict(lr=1e-4, max_epochs=200, patience=10, batch_size=128)
