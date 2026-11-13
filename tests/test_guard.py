"""Tests for budget_guard.guard.BudgetGuard — the wrapper API.

Uses synthetic runs only (no BAGEN data): each run is a vector of per-turn token
totals, from which we build the extractor-schema table the guard consumes.
"""
import numpy as np
import pandas as pd
import pytest

from budget_guard.guard import BudgetGuard, GuardStep


def _runs(n, seed, low=80, high=200, nmin=3, nmax=8):
    """n synthetic runs; each is an array of per-turn (history-excluded) totals."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(low, high, size=int(rng.integers(nmin, nmax))).astype(float)
        for _ in range(n)
    ]


def _logs_from_runs(runs, budget=None):
    """Extractor-schema DataFrame (one row per prefix k=1..N-1) from raw runs."""
    rows = []
    for r, per in enumerate(runs):
        total = float(per.sum())
        N = len(per)
        feasible = True if budget is None else total <= budget
        for k in range(1, N):
            tsf = float(per[:k].sum())
            rows.append(
                {
                    "rollout_index": r,
                    "sample_id": f"r{r}t{k}",
                    "k": k,
                    "N": N,
                    "tokens_so_far": tsf,
                    "total_rollout_tokens": total,
                    "y": total - tsf,
                    "can_finish": feasible,
                    "rollout_success": feasible,
                    "per_turn_total": list(per[:k]),
                }
            )
    return pd.DataFrame(rows)


def test_from_logs_builds_and_steps():
    guard = BudgetGuard.from_logs(_logs_from_runs(_runs(120, 0)), online=False)
    guard.reset()
    step = guard.step(120.0)
    assert isinstance(step, GuardStep)
    assert step.k == 1 and step.spent == 120.0
    assert 0.0 <= step.lo <= step.hi
    assert np.isfinite(step.lo)


def test_guard_covers_on_exchangeable_data():
    """On exchangeable synthetic data the split-CQR interval must hit ~0.90."""
    guard = BudgetGuard.from_logs(_logs_from_runs(_runs(240, 0)), alpha=0.1, online=False)
    covered = []
    for per in _runs(80, 123):
        guard.reset()
        total = float(per.sum())
        for k in range(1, len(per) + 1):
            step = guard.step(per[k - 1])
            if k < len(per):  # prefixes are k=1..N-1
                y_k = total - float(per[:k].sum())
                covered.append(step.lo <= y_k <= step.hi)
    assert 0.82 <= float(np.mean(covered)) <= 0.99


def test_guard_aborts_doomed_run_and_spares_feasible_run():
    guard = BudgetGuard.from_logs(_logs_from_runs(_runs(240, 0)), budget=200.0, online=False)

    guard.reset()  # clearly doomed: 6 x 150 = 900 >> 200
    aborted = any(guard.step(150.0).abort for _ in range(6))
    assert aborted

    guard.budget = 10_000_000.0  # unreachable budget: never abort
    guard.reset()
    assert not any(guard.step(150.0).abort for _ in range(6))


def test_projected_total_lo():
    step = GuardStep(k=2, spent=300.0, lo=50.0, hi=800.0, abort=False)
    assert step.projected_total_lo == 350.0


def test_observe_updates_online_window_and_noop_when_offline():
    online = BudgetGuard.from_logs(_logs_from_runs(_runs(120, 0)), online=True, window=50)
    added = online.observe([100.0, 120.0, 110.0, 130.0])  # 4 turns -> 3 prefixes
    assert added == 3

    offline = BudgetGuard.from_logs(_logs_from_runs(_runs(120, 0)), online=False)
    assert offline.observe([100.0, 120.0, 110.0]) == 0


def test_online_mode_tracks_upward_drift_better_than_frozen():
    """Frozen calibration under-covers when deployment y drifts up; the online
    window, fed each finished run, restores coverage."""
    train = _logs_from_runs(_runs(240, 0, low=80, high=200))
    drifted = _runs(120, 7, low=260, high=520)  # larger per-turn spend at deployment

    def run_stream(online):
        guard = BudgetGuard.from_logs(train, alpha=0.1, online=online, window=100)
        covered = []
        for per in drifted:
            guard.reset()
            total = float(per.sum())
            for k in range(1, len(per) + 1):
                step = guard.step(per[k - 1])
                if k < len(per):
                    covered.append(step.lo <= total - float(per[:k].sum()) <= step.hi)
            guard.observe()
        return float(np.mean(covered))

    assert run_stream(online=True) >= run_stream(online=False)


def test_guard_abort_is_run_level_and_rarely_stops_feasible_runs():
    """The abort uses the run-level lower bound: feasible runs (total <= budget) are
    stopped at most ~abort_delta of the time, even though every turn is checked."""
    budget = 1000.0
    runs = _runs(400, 0, low=80, high=200, nmin=3, nmax=9)
    guard = BudgetGuard.from_logs(_logs_from_runs(runs, budget=budget), budget=budget,
                                  online=False, abort_delta=0.10)
    assert guard.abort_qhat is not None and np.isfinite(guard.abort_qhat)
    stopped = []
    for per in _runs(300, 99, low=80, high=200, nmin=3, nmax=9):
        if per.sum() > budget:
            continue  # only feasible runs count toward false aborts
        guard.reset()
        stopped.append(any(guard.step(t).abort for t in per[:-1]))
    assert len(stopped) > 50
    assert float(np.mean(stopped)) <= 0.20


def test_guard_without_feasibility_labels_falls_back_to_per_turn_rule():
    df = _logs_from_runs(_runs(120, 0)).drop(columns=["can_finish"])
    guard = BudgetGuard.from_logs(df, budget=200.0, online=False)
    assert guard.abort_qhat is None
    guard.reset()
    assert any(guard.step(150.0).abort for _ in range(6))
