"""
extract.py - STAGE 1: record every analysis-split sample's activations.

Paper Eq. 1: for a layer l with N_l units and S samples, the activation matrix is
    X^(l) = [x_1, ..., x_{N_l}]^T  in R^{N_l x S},
where row x_i is unit i's response across the S samples - the ANN analogue of a
brain region's fMRI time series. This stage just records the raw material; pg.py
later picks, per class, which samples form each X^(l).

What is saved (outputs/activations.npz):
    fc1 (N, 300), fc2 (N, 100), fc3 (N, 10)  float32 activations (post-ReLU for fc1/fc2)
    fc1_pre, fc2_pre                         pre-ReLU values (for the as_coded variant only)
    labels (N,), preds (N,), correct (N,) bool, test_idx (N,) index into MNIST test

Compared with min_scripts/activation_ext3.py this:
  * reads the analysis half of the TEST set (the train set has ~no misclassifications),
  * stores post-ReLU activations (see models.py),
  * writes one small .npz (~1 MB) instead of thousands of .pt files that also
    stored the input image (and, via a tensor view, the whole batch - see the
    engineering log on the 2 MB/file mystery).

Run:  python extract.py
"""

import argparse

import numpy as np
import torch

import config
from data import get_data
from models import load_lenet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=config.CKPT_PATH)
    ap.add_argument("--out", default=config.ACTIVATIONS_PATH)
    args = ap.parse_args()

    d = get_data()
    model = load_lenet(args.ckpt)

    # One forward pass over the 5,000 analysis images. Batched only to bound memory.
    acts = {name: [] for name in config.SAVED_ACTS}     # contains activations for each layer, per batch
    preds = []                                          # contains predicted class for each sample, per batch
    with torch.no_grad():
        for start in range(0, len(d["analysis_x"]), 1000):  # batch size 1000
            logits, a = model(d["analysis_x"][start:start + 1000], return_acts=True)
            preds.append(logits.argmax(1))              # argmax returs the predicted class for each sample in the batch
            for name in config.SAVED_ACTS:
                acts[name].append(a[name])

    # Now acts is, for example for the fc1,
    #  acts["fc1"] = [fc1_activations_from_batch0, fc1_activations_from_batch1, ...]
    # preds is,
    #  preds = [preds_from_batch0, preds_from_batch1, ...] ; each item is a 1D tensor of shape (batch size,)

    preds = torch.cat(preds).numpy()                    # concatenate the predictions from all batches
    labels = d["analysis_y"].numpy()
    correct = preds == labels                           # element-wise comparison to get a boolean array indicating correct predictions
    arrays = {name: torch.cat(v).numpy().astype(np.float32) for name, v in acts.items()}

    # Now arrays is, for example for the fc1,
    #  arrays["fc1"] = fc1_activations_from_all_batches ; shape (5000, 300)
    #  arrays["fc2"] = fc2_activations_from_all_batches ; shape (5000, 100) and so on.

    # Pylance flags **arrays as possibly hitting savez_compressed's `allow_pickle: bool`
    # kwarg; the keys are layer names from config.SAVED_ACTS, so that can't happen.
    np.savez_compressed(args.out, labels=labels, preds=preds, correct=correct,
                        test_idx=d["analysis_idx"], **arrays)  # pyright: ignore[reportArgumentType]

    # ── Report: how many samples PG1 will have for each (class, split) ──
    # The "incorrect" column is the binding constraint. The paper's S = 50 is not
    # reachable for misclassified MNIST digits, so pg.py uses min(count, S_MAX).
    print(f"Analysis split: {len(labels)} samples, accuracy {100 * correct.mean():.2f}%")
    print("class | correct | incorrect")
    for c in range(config.NUM_CLASSES):
        m = labels == c
        print(f"  {c}   |  {int((m & correct).sum()):5d}  |  {int((m & ~correct).sum()):4d}")
    for name in config.LAYERS:
        dead = (arrays[name].max(0) <= 0).sum() if name != "fc3" else 0
        print(f"{name}: shape {arrays[name].shape}, units never active on this split: {dead}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
