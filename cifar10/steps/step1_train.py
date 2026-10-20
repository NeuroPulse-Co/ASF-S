"""
Step 1 – train the dense model with the paper's recipe [paper App. C, Table 3].

  Adam (Conv2Net: lr 2e-4, wd 5e-4), batch 128, ≤ 100 epochs,
  ReduceLROnPlateau on val loss (patience 10, factor 0.1),
  early stopping on val loss, checkpoint = lowest val loss.

Gate [paper Table 4]: test accuracy 72.69% ± 1.5 (Conv2Net) / 88.10% (Conv6Net).
Outputs: model_best.pth, history.csv, curves.png, gate.json.
"""

import csv

import torch

from common import C, check, parse_args, seed_everything, step_dir, write_gate
import data
import models
import paper_targets as P
from asfs.metrics import accuracy, ece, predict
from asfs.train import fit

ACC_TOLERANCE = 1.5


def plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for k in ("train_loss", "val_loss"):
        ax[0].plot(ep, [h[k] for h in history], label=k)
    for k in ("train_acc", "val_acc"):
        ax[1].plot(ep, [h[k] for h in history], label=k)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy (%)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    def extra(ap):
        ap.add_argument("--max-epochs", type=int, default=None, help="override Table 3 (for smoke tests)")
        ap.add_argument("--es-patience", type=int, default=C.EARLY_STOP_PATIENCE,
                        help="early-stopping patience; the paper does not state it")
    args = parse_args("step1_train", extra)
    seed_everything()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    out = step_dir(args, "step1_train")
    cfg = C.TRAIN[args.model]
    max_epochs = args.max_epochs or cfg["max_epochs"]

    splits = data.get_splits(args.protocol)
    train_dl = data.loader(splits["train"], cfg["batch_size"], True, device, args.backend)
    val_dl = data.loader(splits["val"], cfg["batch_size"], False, device, args.backend)
    test_dl = data.loader(splits["test"], cfg["batch_size"], False, device, args.backend)
    print(f"train {len(splits['train'])} ({splits['train'].name})  val {len(splits['val'])} "
          f"({splits['val'].name})  test {len(splits['test'])}  device {device}")

    model = models.build(args.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=C.PLATEAU_PATIENCE, factor=C.PLATEAU_FACTOR)
    best, history = fit(model, train_dl, val_dl, opt, max_epochs, args.es_patience,
                        monitor="val_loss", scheduler=sched)

    torch.save(model.state_dict(), out / "model_best.pth")
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader(); w.writerows(history)
    plot(history, out / "curves.png")

    logits, labels = predict(model, test_dl)
    test_acc, test_ece = round(accuracy(logits, labels), 2), round(ece(logits, labels), 4)
    checks = [check("test_acc", test_acc, P.DENSE[args.model]["acc"], tolerance=ACC_TOLERANCE)]
    write_gate(out, "step1_train", args, checks, info={
        "best_epoch": best["epoch"], "epochs_run": len(history), "best_val_loss": round(best["val_loss"], 4),
        "best_val_acc": round(best["val_acc"], 2), "test_ece": test_ece,
        "minutes": round(sum(h["seconds"] for h in history) / 60, 1),
        "val_is_test": splits["val"] is splits["test"], "es_patience": args.es_patience,
        "backend": args.backend,
        "lr_drops_at": [h["epoch"] for p, h in zip(history, history[1:]) if h["lr"] < p["lr"]]})


if __name__ == "__main__":
    main()
