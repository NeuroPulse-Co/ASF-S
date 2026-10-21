"""
test_sanity.py - Checks that each building block does what the paper / this README says.

Each test guards against a failure that is silent in a real run (the numbers still
come out, they just mean something else). Several of these failures actually
happened in the earlier cifar100/ and min_scripts/ code.

Run:  python test_sanity.py        (no pytest needed; ~10 s on CPU)
"""

import numpy as np
import torch

import config
from data import get_data
from importance import score_fc2, score_fc3
from models import count_flops, count_params, load_lenet
from pg import canonical_sign, pg1_as_coded, pg1_paper
from prune import evaluate, prune_fc1

torch.manual_seed(0)
RNG = np.random.default_rng(0)
D = get_data()
BASE = load_lenet(config.CKPT_PATH)


def test_checkpoint_matches_paper():
    """The restored checkpoint must reproduce paper Table 4 (98.51%, 267K params, 0.53 MFLOPs)."""
    x = torch.cat([D["analysis_x"], D["eval_x"]]); y = torch.cat([D["analysis_y"], D["eval_y"]])
    acc, _ = evaluate(BASE, x, y)
    assert abs(acc - 98.51) < 0.01, acc
    assert count_params(BASE) == 266_610 and count_flops(BASE) == 532_400


def test_split_is_disjoint_and_balanced():
    """Analysis and eval halves must not overlap (else 'eval' is not held out)."""
    a, e = set(D["analysis_idx"]), set(D["eval_idx"])
    assert not (a & e) and len(a | e) == 10_000
    counts = np.bincount(D["analysis_y"].numpy()) - np.bincount(D["eval_y"].numpy())
    assert np.abs(counts).max() <= 1, counts


def test_prune_nothing_is_identity():
    """Keeping all 300 neurons must reproduce the original logits exactly."""
    m = prune_fc1(BASE, np.arange(300))
    x = D["eval_x"][:256]
    assert torch.allclose(m(x), BASE(x), atol=1e-6)


def test_pruning_equals_zeroing_the_neurons():
    """THE key correctness property (the one cifar100/rebuild_fc violated): a structurally
    pruned model must equal the original model with the removed neurons silenced."""
    keep = np.sort(RNG.choice(300, 90, replace=False))
    pruned = prune_fc1(BASE, keep)
    silenced = load_lenet(config.CKPT_PATH)
    drop = np.setdiff1d(np.arange(300), keep)
    with torch.no_grad():
        silenced.fc1.weight[drop] = 0
        silenced.fc1.bias[drop] = 0          # relu(0) = 0 -> neuron outputs 0 always
    x = D["eval_x"][:256]
    assert torch.allclose(pruned(x), silenced(x), atol=1e-5)
    assert pruned.fc1.out_features == 90 and pruned.fc2.in_features == 90


def test_pg1_recovers_two_functional_networks():
    """Two groups of units that each co-activate on a different half of the samples are
    two 'functional networks'. PG1 must put them at opposite ends (opposite signs)."""
    S = 40
    X = np.zeros((20, S))
    X[:10, :20] = 1 + 0.1 * RNG.random((10, 20))   # units 0-9 fire on samples 0-19
    X[10:, 20:] = 1 + 0.1 * RNG.random((10, 20))   # units 10-19 fire on samples 20-39
    X[:, :] += 0.2 * RNG.random((20, S))           # background noise keeps the graph connected
    g, evals, active = pg1_paper(X, sigma=config.SIGMA, alpha=config.ALPHA)
    assert active == 20
    assert np.all(np.sign(g[:10]) == np.sign(g[0])) and np.all(np.sign(g[10:]) == -np.sign(g[0]))
    assert (evals > 0.999).sum() == 1              # connected graph -> PG1 is unique


def test_dead_units_do_not_hijack_pg1():
    """An all-zero unit must get PG1 = 0 and must not create a spurious eigenvalue of 1."""
    X = RNG.random((10, 30)); X[3] = 0
    g, evals, active = pg1_paper(X, config.SIGMA, config.ALPHA)
    assert active == 9 and g[3] == 0 and (evals > 0.999).sum() == 1


def test_sign_convention_is_deterministic():
    """v and -v are the same eigenvector; canonical_sign must map both to one vector."""
    g = RNG.standard_normal(50)
    assert np.allclose(canonical_sign(g), canonical_sign(-g))


def test_as_coded_collapses_to_sign_pattern():
    """Documents engineering-log finding #3: with one sample, PG1 is just the sign
    pattern of that sample (two distinct values)."""
    x = RNG.standard_normal(100)
    g, _ = pg1_as_coded(x, config.AS_CODED_SIGMA)
    assert len(np.unique(np.round(g, 6))) == 2
    assert len({tuple(np.sign(np.round(g[x > 0], 6)))}) == 1   # positives share one value


def test_vectorised_scores_match_the_formula():
    """score_fc2 / score_fc3 are vectorised; check them against the per-neuron formula
    s_{j,k} = <W2[:, j], u_k * w_bar3>  and  <W3 @ W2[:, j], u_k>."""
    W2, W3 = BASE.fc2.weight.detach().numpy(), BASE.fc3.weight.detach().numpy()
    u2, u3 = RNG.standard_normal((2, 100)), RNG.standard_normal((2, 10))
    wb = W3.mean(0)
    raw2 = np.array([np.mean([abs(W2[:, j] @ (u * wb)) for u in u2]) for j in range(300)])
    raw3 = np.array([np.mean([abs((W3 @ W2[:, j]) @ u) for u in u3]) for j in range(300)])
    mm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-12)
    assert np.allclose(score_fc2(W2, W3, u2), mm(raw2), atol=1e-6)
    assert np.allclose(score_fc3(W2, W3, u3), mm(raw3), atol=1e-6)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
