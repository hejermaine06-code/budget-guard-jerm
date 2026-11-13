import numpy as np
import pandas as pd
import pytest

from budget_guard.conformal import (
    aci_cqr,
    cqr_split,
    cqr_split_rollout,
    spci,
    split_conformal_absres,
    total_minus_spent,
)
from budget_guard.features import LEAK_COLUMNS, build_xy
from budget_guard.metrics import marginal_coverage, mean_width, rollout_coverage
from budget_guard.split import group_split


def _synthetic(n=2000, seed=0):
    """Exchangeable regression data where split conformal must hit ~0.90."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 3 * X["a"].to_numpy() - 2 * X["b"].to_numpy() + rng.normal(scale=1.0, size=n)
    y = y - y.min()  # nonneg, like remaining tokens
    idx = np.arange(n)
    return X.iloc[idx[:1000]], y[:1000], X.iloc[idx[1000:1500]], y[1000:1500], X.iloc[idx[1500:]], y[1500:]


@pytest.mark.parametrize("method", ["absres", "cqr"])
def test_split_conformal_coverage_on_exchangeable_data(method):
    Xtr, ytr, Xca, yca, Xte, yte = _synthetic()
    if method == "absres":
        lo, hi, _ = split_conformal_absres(Xtr, ytr, Xca, yca, Xte)
    else:
        lo, hi, _ = cqr_split(Xtr, ytr, Xca, yca, Xte)
    cov = marginal_coverage(yte, lo, hi)
    assert 0.85 <= cov <= 0.96, f"{method} coverage {cov}"
    assert mean_width(lo, hi) > 0


@pytest.mark.parametrize("online", [aci_cqr, spci])
def test_online_methods_recover_under_shift(online):
    """Shift the test distribution; online methods must stay near nominal anyway."""
    Xtr, ytr, Xca, yca, Xte, yte = _synthetic()
    yte_shifted = yte * 1.8 + 5.0  # deployment drift
    lo, hi, _ = online(Xtr, ytr, Xca, yca, Xte, yte_shifted)
    cov = marginal_coverage(yte_shifted, lo, hi)
    assert cov >= 0.80, f"online coverage under shift {cov}"


class _ConstModel:
    """Deterministic stand-in for a fitted quantile model."""

    def __init__(self, value: float):
        self._value = value

    def predict(self, X):
        return np.full(len(X), self._value, dtype=float)


@pytest.mark.parametrize("method", ["aci", "spci", "dtaci"])
def test_rollout_safe_revelation_freezes_within_rollout(method, monkeypatch):
    """With ``groups``, predictions inside a rollout are invariant to that rollout's
    own labels — remaining-to-completion labels do not exist until the rollout ends.
    Per-step revelation (groups=None) is NOT invariant, which is exactly the leak
    the rollout-safe schedule closes; asserting both proves the test can see it."""
    import budget_guard.conformal as conformal_mod

    monkeypatch.setattr(
        conformal_mod, "_fit_cqr", lambda X, y, alpha: (_ConstModel(40.0), _ConstModel(60.0))
    )
    fn = {
        "aci": lambda *a, **k: conformal_mod.aci_cqr(*a, gamma=0.05, **k),
        "spci": conformal_mod.spci,
        "dtaci": conformal_mod.dtaci,
    }[method]

    X_cal = pd.DataFrame({"a": np.zeros(10)})
    y_cal = np.linspace(30.0, 70.0, 10)
    n_test = 12
    X_test = pd.DataFrame({"a": np.zeros(n_test)})
    groups = np.repeat([0, 1], 6)
    y_plain = np.full(n_test, 50.0)  # every label comfortably covered
    y_leaky = y_plain.copy()
    y_leaky[1:5] = 1e6  # rollout 0's later labels explode

    lo_a, hi_a, _ = fn(X_cal, y_cal, X_cal, y_cal, X_test, y_plain, alpha=0.1, groups=groups)
    lo_b, hi_b, _ = fn(X_cal, y_cal, X_cal, y_cal, X_test, y_leaky, alpha=0.1, groups=groups)
    # rollout 0 (rows 0-5): its own labels are unobservable mid-rollout -> identical
    assert np.array_equal(lo_a[:6], lo_b[:6]) and np.array_equal(hi_a[:6], hi_b[:6])
    # rollout 1 (rows 6-11): rollout 0 finished, labels legitimately revealed -> reacts
    assert not np.array_equal(hi_a[6:], hi_b[6:])

    # positive control: per-step revelation leaks same-rollout labels into rows 2-5
    _, hi_c, _ = fn(X_cal, y_cal, X_cal, y_cal, X_test, y_plain, alpha=0.1)
    _, hi_d, _ = fn(X_cal, y_cal, X_cal, y_cal, X_test, y_leaky, alpha=0.1)
    assert not np.array_equal(hi_c[2:6], hi_d[2:6])


def test_group_split_partitions_by_rollout():
    df = pd.DataFrame({"rollout_index": np.repeat(np.arange(40), 5)})
    masks = group_split(df, seed=1)
    total = sum(m.astype(int) for m in masks.values())
    assert (total == 1).all()
    for name, mask in masks.items():
        rollouts = set(df.loc[mask, "rollout_index"])
        for other, omask in masks.items():
            if other != name:
                assert rollouts.isdisjoint(set(df.loc[omask, "rollout_index"]))


def test_features_are_leak_free_and_finite():
    df = pd.DataFrame(
        {
            "rollout_index": [0, 0, 1],
            "k": [1, 2, 1],
            "N": [3, 3, 2],
            "tokens_so_far": [100, 260, 80],
            "total_rollout_tokens": [400, 400, 200],
            "y": [300, 140, 120],
            "can_finish": [True, True, False],
            "rollout_success": [True, True, False],
            "sample_id": ["r0t1", "r0t2", "r1t1"],
            "per_turn_total": [[100], [100, 160], [80]],
        }
    )
    X, y, groups = build_xy(df)
    assert not (set(X.columns) & set(LEAK_COLUMNS))
    assert np.isfinite(X.to_numpy()).all()
    assert (y == df["y"].to_numpy()).all()
    assert (groups == df["rollout_index"].to_numpy()).all()


def _synthetic_rollouts(n_rollouts=1000, prefixes=5, seed=0):
    """Rollouts whose prefixes share a rollout-level shock, so they are dependent."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_rollouts), prefixes)
    shock = rng.normal(scale=1.5, size=n_rollouts)[groups]
    X = pd.DataFrame({"a": rng.normal(size=len(groups)), "k": np.tile(np.arange(prefixes), n_rollouts)})
    y = 3 * X["a"].to_numpy() + shock + rng.normal(scale=0.3, size=len(groups))
    return X, y - y.min(), groups


def test_rollout_cqr_covers_whole_runs():
    """One score per calibration rollout -> ~1-alpha of test rollouts covered at every prefix."""
    X, y, g = _synthetic_rollouts()
    tr, ca, te = g < 500, (g >= 500) & (g < 750), g >= 750
    lo, hi, info = cqr_split_rollout(X[tr], y[tr], X[ca], y[ca], g[ca], X[te])
    assert info["n_cal_rollouts"] == 250
    run_cov = rollout_coverage(y[te], lo, hi, g[te])
    assert run_cov >= 0.84, f"rollout-level coverage {run_cov}"
    # covering whole runs is stricter, so per-prefix coverage should sit above nominal
    assert marginal_coverage(y[te], lo, hi) >= 0.90


def test_rollout_cqr_rejects_mismatched_groups():
    X, y, g = _synthetic_rollouts(n_rollouts=50)
    with pytest.raises(ValueError):
        cqr_split_rollout(X[:100], y[:100], X[100:200], y[100:200], g[100:150], X[200:])


def test_total_minus_spent_is_calibrated_on_exchangeable_runs():
    """Runs with random totals, observed at several prefixes: the arithmetic rule,
    conformalized like split-CQR, should land near nominal per-prefix coverage."""
    rng = np.random.default_rng(1)
    n_runs, k = 1200, 4
    totals = rng.gamma(shape=4.0, scale=500.0, size=n_runs)
    g = np.repeat(np.arange(n_runs), k)
    frac = np.tile(np.arange(1, k + 1) / (k + 1), n_runs)
    spent = totals[g] * frac
    y = totals[g] - spent
    tr, ca, te = g < 600, (g >= 600) & (g < 900), g >= 900
    lo, hi, info = total_minus_spent(spent[tr], y[tr], g[tr], spent[ca], y[ca], spent[te])
    assert info["total_lo"] < info["total_hi"]
    assert (lo >= 0).all() and (hi >= lo).all()
    assert 0.85 <= marginal_coverage(y[te], lo, hi) <= 0.96


def test_rollout_cqr_with_uneven_run_lengths_and_run_effects():
    """Code review #9: runs of very different lengths, longer runs harder (a shared
    run-level shock that grows with length). One score per calibration run must still
    cover ~1-alpha of whole test runs."""
    rng = np.random.default_rng(3)
    lengths = rng.integers(1, 30, size=1200)
    g = np.repeat(np.arange(len(lengths)), lengths)
    shock = (rng.normal(size=len(lengths)) * (0.5 + lengths / 10))[g]
    X = pd.DataFrame({"a": rng.normal(size=len(g)), "k": np.concatenate([np.arange(n) for n in lengths])})
    y = 2 * X["a"].to_numpy() + shock + rng.normal(scale=0.3, size=len(g))
    y = y - y.min()
    tr, ca, te = g < 600, (g >= 600) & (g < 900), g >= 900
    lo, hi, _ = cqr_split_rollout(X[tr], y[tr], X[ca], y[ca], g[ca], X[te])
    assert rollout_coverage(y[te], lo, hi, g[te]) >= 0.84


def test_windowed_cqr_runs_window_counts_runs():
    from budget_guard.conformal import windowed_cqr_runs

    X, y, g = _synthetic_rollouts(n_rollouts=300, prefixes=5, seed=4)
    tr, ca, te = g < 150, (g >= 150) & (g < 225), g >= 225
    lo, hi, info = windowed_cqr_runs(X[tr], y[tr], X[ca], y[ca], X[te], y[te], g[ca], g[te],
                                     window_runs=10)
    assert info["window_runs"] == 10
    assert (lo >= 0).all() and (hi >= lo).all()
    # every prefix of one test run shares one qhat (labels only after the run ends)
    first = g[te] == g[te][0]
    widths = (hi - lo)[first]
    assert 0.80 <= marginal_coverage(y[te], lo, hi) <= 0.99
    assert np.isfinite(widths).all()
