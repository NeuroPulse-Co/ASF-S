"""
scores – filter-importance criteria. Every scorer has the same signature:

    scorer(model, ctx) -> {conv name: np.ndarray (F,) float}     # higher = more important = keep

`model` is a dense model in eval mode (models.Conv2Net / Conv6Net, CONV_SEQ order). `ctx` is a ScoreContext.
Scorers must not modify `model` (work on a deepcopy if they need hooks/grads) and must be deterministic
given ctx.seed. Register with @register("name"); look up with get(name) / names().
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

REGISTRY: dict = {}


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
    for mod in ("pgi", "weight", "data_driven"):
        try:
            importlib.import_module(f"{__name__}.{mod}")
        except ModuleNotFoundError as e:
            if e.name != f"{__name__}.{mod}":
                raise
