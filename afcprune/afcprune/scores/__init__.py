"""
scores – filter-importance criteria. Every scorer has the same signature:

    scorer(model, ctx) -> {conv name: np.ndarray (F,) float}     # higher = more important = keep

`model` is a dense model in eval mode (models.Conv2Net / Conv6Net, CONV_SEQ order). `ctx` is a ScoreContext.
Scorers must not modify `model` (work on a deepcopy if they need hooks/grads) and must be deterministic
given ctx.seed. Register with @register("name"); look up with get(name) / names().

Weight-level scorers (M1; the proposal masks individual weights) use a second registry:

    wscorer(model, ctx) -> {conv name: np.ndarray shaped like getattr(model, conv).weight}   # higher = keep

Only the backbone convs (model.CONV_SEQ) are scored: in TIL every task has its own output head, which is refit,
not masked. Register with @wregister("name"); look up with wget(name) / wnames(). unit_sum(wscores) turns a
weight-level score into a per-filter one (sum over the filter's weights) for the unit-level harness.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

REGISTRY: dict = {}
WREGISTRY: dict = {}


@dataclass
class ScoreContext:
    device: Any = "cpu"
    calib: Optional[Iterable] = None   # iterable of normalised (x, y) batches, NO augmentation (train split)
    seed: int = 42
    extra: dict = field(default_factory=dict)   # e.g. {"components": (K, C) PCA comps} for PGI


def register(name: str) -> Callable:
    def deco(fn):
        if name in REGISTRY:
            raise KeyError(f"scorer {name!r} registered twice")
        REGISTRY[name] = fn
        return fn
    return deco


def wregister(name: str) -> Callable:
    def deco(fn):
        if name in WREGISTRY:
            raise KeyError(f"weight scorer {name!r} registered twice")
        WREGISTRY[name] = fn
        return fn
    return deco


def wget(name: str) -> Callable:
    _load_all()
    return WREGISTRY[name]


def wnames():
    _load_all()
    return sorted(WREGISTRY)


def unit_sum(wscores):
    """{conv: (F, ...)} weight scores → {conv: (F,)} per-filter sums."""
    return {c: np.asarray(v).reshape(len(v), -1).sum(1) for c, v in wscores.items()}


def get(name: str) -> Callable:
    _load_all()
    return REGISTRY[name]


def names():
    _load_all()
    return sorted(REGISTRY)


def conv_names(model):
    return [c for c, _, _ in model.CONV_SEQ]


def _load_all():
    # import modules that register scorers (each module registers on import)
    import importlib
    for mod in ("pgi", "weight", "data_driven", "wlevel", "gradsvd", "sfc", "knockout"):
        try:
            importlib.import_module(f"{__name__}.{mod}")
        except ModuleNotFoundError as e:
            if e.name != f"{__name__}.{mod}":
                raise
