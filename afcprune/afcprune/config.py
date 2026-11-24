"""
config.py – every default of afcprune.

Values the seed paper states are marked [paper §...]; values it leaves open are marked [assumed] with their
source. Paper numbers used as targets live in paper_targets.py. Scripts override these from the CLI.
"""

import os
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = Path(os.environ.get("AFCP_DATA", ROOT / "data")).expanduser()
OUTPUTS_DIR = Path(os.environ.get("AFCP_OUTPUTS", ROOT / "outputs")).expanduser()

SEED        = 42
NUM_CLASSES = 10
CLASSES     = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# ── Data [paper App. C] ───────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)
JITTER   = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
ROTATION = 15.0
CROP_PAD = 4
AUGMENT_BACKEND = "gpu"
NUM_WORKERS     = 4

# ── Protocol ──────────────────────────────────────────────────
# clean: 45k train / 5k stratified val (split seed fixed, independent of the model seed); test only via
#        protocol.allow_test(). paper: val = test, as the original scripts (reproduction rows only).
PROTOCOL   = "clean"
VAL_SIZE   = 5000
SPLIT_SEED = 42
ACT_PER_CLASS = 1000      # [assumed] paper protocol PG set: 1000/class train subset

# ── Dense training [paper App. C, Table 3] ────────────────────
TRAIN = {
    "conv2net": dict(lr=2e-4, weight_decay=5e-4, batch_size=128, max_epochs=100),
    "conv6net": dict(lr=3e-4, weight_decay=5e-4, batch_size=128, max_epochs=100),
}
PLATEAU_PATIENCE    = 10
PLATEAU_FACTOR      = 0.1
EARLY_STOP_PATIENCE = 15  # [assumed]

# ── AFC / PG1 [paper §3–4] ────────────────────────────────────
S_SAMPLES = 50            # [assumed] S in Eq. 1 (original max_samples)
SIGMA     = "median"      # [assumed] median rule; σ = 0.1 disconnects W (cifar10 LOG, step 3–4)
ALPHA     = 0.5           # [assumed] Eq. 6
EPS       = 1e-6          # [assumed] Eq. 3
PCA_K     = 4             # [paper §C.3]
SUBSPACE_K = 3            # non-trivial diffusion eigenvectors compared by fidelity (P3: not v2 alone)

# ── Recovery ──────────────────────────────────────────────────
FINETUNE = dict(lr=1e-4, max_epochs=200, patience=10, batch_size=128)   # [assumed] original recipe
RIDGE_LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]                     # chosen on val, same for all selectors
BN_RECAL_BATCHES = 50                                                   # train batches (no augmentation)

# ── Budgets: total params↓ % (P8) ─────────────────────────────
BUDGETS = [50.0, 70.0, 75.8, 85.0, 90.0]
