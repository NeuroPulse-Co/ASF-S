"""
config.py - Central configuration for the ASF-S / LeNet-300-100 / MNIST experiment.

Every stage script imports its paths and defaults from here, so there is exactly
one place to look when asking "which file does this read?" or "what sigma was used?".
Individual scripts can still override most values from the command line.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(HERE, "data")            # torchvision downloads MNIST here (gitignored)
CKPT_PATH = os.path.join(HERE, "checkpoints", "lenet300_100_best.pth")
OUT_DIR = os.path.join(HERE, "outputs")          # intermediate artefacts (gitignored)
RESULTS_DIR = os.path.join(HERE, "results")      # final CSVs / plots (small, committable)

ACTIVATIONS_PATH = os.path.join(OUT_DIR, "activations.npz")  # written by extract.py
PG_DIR = os.path.join(OUT_DIR, "pg")                          # written by pg.py
IMPORTANCE_DIR = os.path.join(OUT_DIR, "importance")          # written by importance.py

# ── Dataset ───────────────────────────────────────────────────────────────────
# Standard MNIST statistics (paper App. C: mu = 0.1307, sigma = 0.3081).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
NUM_CLASSES = 10

# The MNIST test set (10,000 images) is split in half, stratified by class:
#   * "analysis" half -> used to build PG1 (needs misclassified samples) and for
#                        early stopping during fine-tuning.
#   * "eval" half     -> touched ONLY to report accuracy / ECE.
# Why not use the training set? The checkpoint is 99.97% accurate on it, so there
# are essentially no misclassified training images to build PG1_incorrect from.
SPLIT_SEED = 0

SEED = 42

# ── Model ─────────────────────────────────────────────────────────────────────
LAYERS = ("fc1", "fc2", "fc3")   # fc3 is the output (logit) layer
LAYER_UNITS = {"fc1": 300, "fc2": 100, "fc3": 10}
# Everything extract.py saves: post-ReLU fc1/fc2, logits fc3, and pre-ReLU fc1/fc2
# (the "_pre" arrays exist only to reproduce the original code in the as_coded variant).
SAVED_ACTS = ("fc1", "fc2", "fc3", "fc1_pre", "fc2_pre")

# ── PG1 extraction (paper Sec. 3-4) ───────────────────────────────────────────
S_MAX = 50        # max samples per (class, correct/incorrect) group. Paper / code used 50.
# Gaussian kernel width in Eq. 3. The original code used 0.1; on these post-ReLU
# activations that leaves the fc2 unit graph split into 15-22 disconnected pieces
# (eigenvalue 1 repeated 15-22 times), which makes "the second eigenvector" - PG1 -
# non-unique and arbitrary. A scan over sigma (see mnist/README.md) showed 0.5 is the
# smallest value that yields ONE connected graph for both fc2 and fc3, with a clear
# spectral gap for fc3. pg.py prints the diagnostics so this can be re-checked.
SIGMA = 0.5
AS_CODED_SIGMA = 0.1   # the as_coded variant keeps the original value, by definition
ALPHA = 0.5       # diffusion-map normalisation exponent in Eq. 6 (same as original code).

# ── Importance (paper Eq. 11-13) ──────────────────────────────────────────────
PCA_COMPONENTS = 4    # paper App. C.3: "top 4 components"

# ── Pruning / fine-tuning ─────────────────────────────────────────────────────
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # fraction of fc1 neurons removed
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FT_LR = 1e-4          # same fine-tuning LR as cifar100/ and min_scripts/
FT_EPOCHS = 20
FT_PATIENCE = 3
FT_BATCH = 128

for _d in (DATA_DIR, OUT_DIR, RESULTS_DIR, PG_DIR, IMPORTANCE_DIR):
    os.makedirs(_d, exist_ok=True)
