"""
Step 2 – output-layer activations for PG1 [paper §3 Eq. 1, §4].

Runs the step-1 model over the PG image set without augmentation and saves, per image:
the output-layer activations (Conv2Net: the 10 `fc` logits, as the original
activation_ext3.py hooked), the label, the prediction and whether it was correct.

  paper protocol: 1000 images per class from the training set (original scripts)
  clean protocol: the 5k validation split

Gate: every (class, correct|incorrect) group has at least S = 50 samples, the number
step 4 stacks into X^(ℓ) (Eq. 1).
Outputs: activations.npz (acts, labels, preds, correct), counts.csv, gate.json.
"""

import csv

import numpy as np
import torch

from common import C, check, parse_args, run_dir, seed_everything, step_dir, write_gate
import data
import models


@torch.no_grad()
def extract(model, loader):
    """Returns numpy (acts (N, units), labels (N,), preds (N,)) for the model's output layer."""
    model.eval()
    acts, labels = [], []
    for x, y in loader:
        acts.append(model(x).float().cpu()); labels.append(y.cpu())
    acts, labels = torch.cat(acts).numpy(), torch.cat(labels).numpy()
    return acts, labels, acts.argmax(1)


def group_counts(labels, correct, num_classes=C.NUM_CLASSES):
    """(num_classes, 2) array: column 0 = correct, column 1 = incorrect."""
    return np.stack([np.bincount(labels[correct], minlength=num_classes),
                     np.bincount(labels[~correct], minlength=num_classes)], 1)


def main():
    args = parse_args("step2_activations")
    seed_everything(args.seed)
    device = torch.device(args.device)
    out = step_dir(args, "step2_activations")

    ckpt = run_dir(args) / "step1_train" / "model_best.pth"
    model = models.build(args.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    split = data.get_splits(args.protocol)["pg"]
    dl = data.loader(split, 500, False, device, args.backend, args.seed)
    acts, labels, preds = extract(model, dl)
    correct = preds == labels
    np.savez_compressed(out / "activations.npz", acts=acts, labels=labels, preds=preds, correct=correct)

    counts = group_counts(labels, correct)
    with open(out / "counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "name", "correct", "incorrect"])
        for c in range(C.NUM_CLASSES):
            w.writerow([c, C.CLASSES[c], *counts[c]])
    print(f"{split.name}: {len(labels)} images, {acts.shape[1]} output units, "
          f"accuracy {100 * correct.mean():.2f}%")
    for c in range(C.NUM_CLASSES):
        print(f"  {c} {C.CLASSES[c]:11s} correct {counts[c, 0]:4d}  incorrect {counts[c, 1]:4d}")

    checks = [check(f"min samples per (class, split) ≥ S={C.S_SAMPLES}", int(counts.min()), C.S_SAMPLES,
                    ok=counts.min() >= C.S_SAMPLES)]
    write_gate(out, "step2_activations", args, checks, info={
        "pg_set": split.name, "n_images": int(len(labels)), "units": int(acts.shape[1]),
        "accuracy": round(100 * float(correct.mean()), 2),
        "min_incorrect": int(counts[:, 1].min()), "min_incorrect_class": C.CLASSES[int(counts[:, 1].argmin())]})


if __name__ == "__main__":
    main()
