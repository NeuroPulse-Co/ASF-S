"""
common.py – shared CLI, run directories and gate reporting for the step scripts.

Every step writes outputs/<run_id>/stepN_<name>/ and ends with gate.json:
  {"step": ..., "protocol": ..., "pass": bool, "checks": [{metric, ours, paper, tolerance, pass}]}
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402


def parse_args(step_name, extra=None):
    ap = argparse.ArgumentParser(description=step_name)
    ap.add_argument("--run-id", default="run01", help="outputs/<run-id>/ holds every step's output")
    ap.add_argument("--model", default="conv2net", choices=["conv2net", "conv6net"])
    ap.add_argument("--protocol", default=C.PROTOCOL, choices=["paper", "clean"])
    ap.add_argument("--backend", default=C.AUGMENT_BACKEND, choices=["gpu", "torchvision"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=C.SEED, help="weights init, shuffling and augmentation")
    if extra:
        extra(ap)
    return ap.parse_args()


def seed_everything(seed=C.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_dir(args):
    """outputs/<run_id>/<model>_<protocol>/ – shared by all steps of one run."""
    d = C.OUTPUTS_DIR / args.run_id / f"{args.model}_{args.protocol}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def step_dir(args, step):
    d = run_dir(args) / step
    d.mkdir(parents=True, exist_ok=True)
    return d


def check(metric, ours, paper, tolerance=None, ok=None):
    """One gate line. Pass = |ours - paper| <= tolerance, unless ok is given explicitly."""
    if ok is None:
        ok = abs(ours - paper) <= tolerance
    return dict(metric=metric, ours=ours, paper=paper, tolerance=tolerance, pass_=bool(ok))


def write_gate(out_dir, step, args, checks, info=None):
    gate = {"step": step, "model": args.model, "protocol": args.protocol,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pass": all(c["pass_"] for c in checks),
            "checks": [{**{k: v for k, v in c.items() if k != "pass_"}, "pass": c["pass_"]}
                       for c in checks],
            "info": info or {}}
    (out_dir / "gate.json").write_text(json.dumps(gate, indent=2))
    print(f"\n── gate: {step} ──")
    for c in gate["checks"]:
        tol = "" if c["tolerance"] is None else f" ± {c['tolerance']}"
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['metric']}: ours {c['ours']}  "
              f"paper {c['paper']}{tol}")
    print(f"  => {'PASS' if gate['pass'] else 'FAIL'}   ({out_dir / 'gate.json'})")
    return gate
