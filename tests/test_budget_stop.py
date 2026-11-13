import numpy as np
import pandas as pd

from budget_guard.economic import _run_rule
from budget_guard.run_budget_stop import budget_trigger_tokens, doomed_anatomy


def _toy_runs(budget=100.0):
    """Three rollouts: a feasible one (ends under budget), a doomed one that blows the
    budget at turn 3, and a doomed one that fails while still under budget."""
    rows = []
    spec = [(0, [30, 60, 90], True),     # feasible, total 90 <= budget
            (1, [40, 80, 120, 160], False),  # doomed, over budget from turn 3
            (2, [20, 40, 60], False)]    # doomed, never over budget (wrong answer)
    for rid, cum, feasible in spec:
        for k, spent in enumerate(cum, start=1):
            rows.append({"rollout_index": rid, "k": k, "tokens_so_far": float(spent),
                         "total_rollout_tokens": float(cum[-1]), "can_finish": feasible})
    return pd.DataFrame(rows)


def test_budget_stop_never_aborts_feasible_runs():
    df = _toy_runs()
    trig = budget_trigger_tokens(df, 100.0)
    out = _run_rule(df, trig)
    assert out["false_abort_rate"] == 0.0
    # doomed spend = 160 + 60; rollout 1 stopped at 120 -> saves 40
    assert np.isclose(out["doomed_savings_frac"], 40 / 220)


def test_doomed_anatomy_counts_only_over_budget_failures():
    df = _toy_runs()
    out = doomed_anatomy(df, budget_trigger_tokens(df, 100.0))
    assert out["n_doomed"] == 2
    assert out["doomed_over_budget_frac"] == 0.5
