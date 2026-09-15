#!/usr/bin/env python3
"""
select_model.py — switch cifar100/ between conv2net, conv6net and vgg16.

The pipeline selects its model by commenting blocks in and out across five files.
This performs exactly the edits you would make by hand.

Idempotent: re-running with the same model is a no-op, and switching models works
from any starting state. Every block must be found exactly once or the script
aborts without writing, so a drifted repo fails loudly instead of being half-edited.

Touches only the model-selection lines. No algorithm code is modified.

    python scripts/select_model.py conv2net
    python scripts/select_model.py --check          # report current state only
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = ["conv2net", "conv6net", "vgg16"]

# ── Multi-line blocks: explicit active / commented forms ──────
# train_cifar100.py's three optimiser blocks. Note the conv2net and conv6net
# `sch = ...` lines are byte-identical, so these must be matched as whole blocks
# rather than line by line.
TRAIN = {
    "conv2net": (
        'm   = Conv2Net(NUM_CLASSES).to(device)\n'
        'opt = optim.Adam(m.parameters(), lr=2e-4, weight_decay=5e-4)\n'
        'sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.1)\n'
        'train(m, opt, sch, num_epochs=200, patience=15, model_name="conv2net")',
    ),
    "conv6net": (
        'm   = Conv6Net(NUM_CLASSES).to(device)\n'
        'opt = optim.Adam(m.parameters(), lr=3e-4, weight_decay=5e-4)\n'
        'sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.1)\n'
        'train(m, opt, sch, num_epochs=200, patience=15, model_name="conv6net")',
    ),
    "vgg16": (
        'm   = VGG16CIFAR(NUM_CLASSES).to(device)\n'
        'opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)\n'
        '# cosine schedule steps every epoch; pass None to train() and step manually\n'
        'sch_cos = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)\n'
        'train(m, opt, scheduler=None, num_epochs=200, patience=20, model_name="vgg16")',
    ),
}

# ── Single-line dict entries, given in their active form ──────
SINGLE = {
    "cifar100/activation_ext_cifar100.py": {
        "conv2net": '    "conv2net": (Conv2Net(NUM_CLASSES),   LAYERS["conv2net"], "conv2net_best.pth"),',
        "conv6net": '    "conv6net": (Conv6Net(NUM_CLASSES),   LAYERS["conv6net"], "conv6net_best.pth"),',
        "vgg16":    '    "vgg16":    (VGG16CIFAR(NUM_CLASSES), LAYERS["vgg16"],    "vgg16_best.pth"),',
    },
    "cifar100/cosine_sim_ext_cifar100.py": {
        "conv2net": '    "conv2net": LAYERS["conv2net"],',
        "conv6net": '    "conv6net": LAYERS["conv6net"],',
        "vgg16":    '    "vgg16":    LAYERS["vgg16"],',
    },
    "cifar100/pg_ext_cifar100.py": {
        "conv2net": '    "conv2net": [PG_FC_LAYER["conv2net"]],',
        "conv6net": '    "conv6net": [PG_FC_LAYER["conv6net"]],',
        "vgg16":    '    "vgg16":    [PG_FC_LAYER["vgg16"]],',
    },
}

# ── `for mn in [...]` loops ───────────────────────────────────
LOOPS = ["cifar100/pg_pruning_cifar100.py", "cifar100/hrank_cifar100.py"]
LOOP_RE = re.compile(r'^for mn in \[.*\]:$', re.M)


def comment_block(text: str) -> str:
    """Prefix each line with '# ', leaving lines that are already comments alone."""
    out = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        out.append(line if stripped.startswith("#") else f"{indent}# {stripped}")
    return "\n".join(out)


def apply(model: str, check_only: bool = False) -> bool:
    """Returns True if every file is already in the requested state."""
    edits = []          # (path, old_text, new_text)
    already = True

    # train_cifar100.py
    path = REPO / "cifar100/train_cifar100.py"
    src = path.read_text()
    for name, (active,) in TRAIN.items():
        commented = comment_block(active)
        want = active if name == model else commented
        have_active = src.count(active) == 1
        have_comment = src.count(commented) == 1
        if not (have_active ^ have_comment):
            sys.exit(f"ERROR: {path.name}: the {name} block is neither cleanly "
                     f"active nor cleanly commented — refusing to edit.")
        current = active if have_active else commented
        if current != want:
            already = False
            edits.append((path, current, want))
            src = src.replace(current, want)

    # single-line dict entries
    for rel, variants in SINGLE.items():
        path = REPO / rel
        src = path.read_text()
        for name, active in variants.items():
            commented = comment_block(active)
            want = active if name == model else commented
            have_active = src.count(active + "\n") == 1
            have_comment = src.count(commented + "\n") == 1
            if not (have_active ^ have_comment):
                sys.exit(f"ERROR: {rel}: the {name} entry is neither cleanly "
                         f"active nor cleanly commented — refusing to edit.")
            current = active if have_active else commented
            if current != want:
                already = False
                edits.append((path, current + "\n", want + "\n"))
                src = src.replace(current + "\n", want + "\n")

    # `for mn in [...]` loops
    for rel in LOOPS:
        path = REPO / rel
        src = path.read_text()
        found = LOOP_RE.findall(src)
        if len(found) != 1:
            sys.exit(f"ERROR: {rel}: expected exactly one 'for mn in [...]:' line, "
                     f"found {len(found)}.")
        want = f'for mn in ["{model}"]:'
        if found[0] != want:
            already = False
            edits.append((path, found[0], want))

    if check_only:
        return already

    # Apply grouped by file so each is written once.
    by_file = {}
    for path, old, new in edits:
        by_file.setdefault(path, []).append((old, new))
    for path, changes in by_file.items():
        src = path.read_text()
        for old, new in changes:
            assert src.count(old) == 1, f"{path}: '{old[:50]}' not unique at write time"
            src = src.replace(old, new)
        path.write_text(src)
        print(f"  patched {path.relative_to(REPO)}")

    return already


def current_model() -> str:
    for m in MODELS:
        if apply(m, check_only=True):
            return m
    return "mixed/unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", choices=MODELS)
    p.add_argument("--check", action="store_true", help="report state, change nothing")
    args = p.parse_args()

    if args.check or not args.model:
        print(f"current selection: {current_model()}")
        return

    if apply(args.model, check_only=True):
        print(f"already set to {args.model} — nothing to do")
        return

    print(f"selecting {args.model}:")
    apply(args.model)
    print(f"done — now set to {current_model()}")


if __name__ == "__main__":
    main()
