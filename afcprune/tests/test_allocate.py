import numpy as np
import pytest

from afcprune import allocate, models


@pytest.fixture(scope="module")
def m2():
    return models.build("conv2net")


def test_table5_point_is_a_candidate(m2):
    c = allocate.candidates(m2, 75.8, tol=0.1)
    assert any(k == {"conv1": 19, "conv2": 16} for k, _ in c)
    assert all(abs(r - 75.8) <= 0.1 for _, r in c)


def test_choose_uses_criterion_only(m2):
    cands = allocate.candidates(m2, 70.0, tol=0.3)
    scores = {"conv1": np.arange(32.0), "conv2": np.arange(64.0)}
    # criterion prefers keeping more conv1 filters
    kept, r, table = allocate.choose(scores, cands, lambda keep: keep["conv1"].sum())
    assert kept["conv1"] == max(k["conv1"] for k, _ in cands)
    assert len(table) == len(cands)


def test_keep_from_counts_top_k():
    keep = allocate.keep_from_counts({"a": np.array([3., 1., 2.])}, {"a": 2})
    assert keep["a"].tolist() == [True, False, True]


def test_deep_model_family_nonempty():
    m6 = models.build("conv6net")
    c = allocate.candidates(m6, 67.0, tol=2.0)
    assert len(c) > 0


@pytest.mark.parametrize("name", ["conv2net", "conv6net"])
def test_analytic_count_matches_surgery(name):
    from afcprune import metrics, surgery
    m = models.build(name)
    rng = np.random.default_rng(0)
    for _ in range(5):
        kept = {c: int(rng.integers(1, getattr(m, c).out_channels + 1)) for c, _, _ in m.CONV_SEQ}
        keep = {c: np.arange(getattr(m, c).out_channels) < kept[c] for c in kept}
        assert allocate.count_params_kept(m, kept) == metrics.count_params(surgery.prune(m, keep))
