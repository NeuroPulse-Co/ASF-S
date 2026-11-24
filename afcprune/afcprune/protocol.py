"""
protocol.py – data splits with a test-set guard (plan P4).

Under the clean protocol method development sees only train (45k) and val (5k). The test split can be read
only inside `with allow_test(reason):`, and every access is appended to <outputs>/test_access.jsonl
(time, reason, caller), so the number of looks at the test set is auditable.

    splits = get_splits("clean")
    splits.train, splits.val, splits.pg          # free
    splits.test                                  # raises TestAccessError
    with allow_test("S5 final scorecard"):
        splits.test                              # allowed, logged

The paper protocol (val = test) is for reproduction rows only; reading .val there counts as test access too.
"""

import contextlib
import json
import time
import traceback
from pathlib import Path

from . import config as C
from . import data

_ALLOWED: list = []


class TestAccessError(RuntimeError):
    pass


@contextlib.contextmanager
def allow_test(reason: str, log_dir: Path = None):
    if not reason:
        raise ValueError("allow_test needs a reason")
    _ALLOWED.append((reason, Path(log_dir or C.OUTPUTS_DIR)))
    try:
        yield
    finally:
        _ALLOWED.pop()


def _log(reason, log_dir, what):
    log_dir.mkdir(parents=True, exist_ok=True)
    caller = traceback.extract_stack(limit=4)[0]
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), reason=reason, split=what,
               caller=f"{Path(caller.filename).name}:{caller.lineno}")
    with open(log_dir / "test_access.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


class Splits:
    def __init__(self, protocol, d):
        self.protocol = protocol
        self._d = d

    def _guarded(self, what):
        if not _ALLOWED:
            raise TestAccessError(f"{what} is (or contains) the CIFAR-10 test set under protocol "
                                  f"{self.protocol!r}; wrap the access in protocol.allow_test(reason)")
        reason, log_dir = _ALLOWED[-1]
        _log(reason, log_dir, what)
        return self._d[what]

    @property
    def train(self):
        return self._d["train"]

    @property
    def pg(self):
        return self._d["pg"]

    @property
    def val(self):
        return self._guarded("val") if self.protocol == "paper" else self._d["val"]

    @property
    def test(self):
        return self._guarded("test")


def get_splits(protocol=C.PROTOCOL, _loaded=None):
    """_loaded: inject a {train, val, test, pg} dict (tests); default loads CIFAR-10 via data.get_splits."""
    return Splits(protocol, _loaded if _loaded is not None else data.get_splits(protocol))
