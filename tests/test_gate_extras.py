import json

import numpy as np
import pandas as pd

from budget_guard.gate_baselines import (
    agent_trigger,
    auc,
    burn_rate_slack_tokens,
    burn_rate_slack_warehouse,
    load_agent_impossible,
    logistic_gate,
    typical_turns,
)
from budget_guard.run_agent_bias import summarize_record
from budget_guard.run_multiresource_overlap import overlap_stats


def _toy():
    rows = []
    for rid, cum in [(0, [100, 200, 300]), (1, [300, 600])]:
        for k, spent in enumerate(cum, start=1):
            rows.append({"rollout_index": rid, "k": k, "tokens_so_far": float(spent),
                         "sample_id": f"rollout-{rid}-turn-{k}"})
    return pd.DataFrame(rows)


def test_typical_turns_counts_usable_turns():
    # rollouts with max k = 3 and 2 -> N = 4 and 3 -> median 3.5
    assert typical_turns(_toy()) == 3.5


def test_burn_rate_slack_projects_current_rate():
    df = _toy()
    s = burn_rate_slack_tokens(df, budget=1000.0, n_turns=4.0)
    # rollout 0, k=1: 100 + 100*3 = 400 -> slack 0.6; rollout 1, k=2: 600 + 300*2 = 1200 -> -0.2
    assert np.isclose(s[0], 0.6) and np.isclose(s[4], -0.2)
    # at k >= n_turns it still projects one more turn
    assert np.isclose(burn_rate_slack_tokens(df.iloc[[2]], 1000.0, 2.0)[0], (1000 - 400) / 1000)


def test_burn_rate_slack_warehouse_takes_worst_resource():
    df = pd.DataFrame({
        "completed_weeks": [5.0], "budget_time_weeks": [10.0],
        "spent_time_weeks": [5.0], "budget_warehouse_item_weeks": [100.0],
        "spent_warehouse_item_weeks": [40.0], "budget_cost_usd": [1000.0],
        "spent_cost_usd": [600.0],
    })
    # time 0.5; capacity 40 + 8*5 = 80 -> 0.2; cost 600 + 120*5 = 1200 -> -0.2
    assert np.isclose(burn_rate_slack_warehouse(df)[0], -0.2)


def test_agent_flags_load_and_align(tmp_path):
    rec = {"results": [
        {"sample_id": "rollout-0-turn-2", "prediction": {"is_impossible": True}},
        {"sample_id": "rollout-0-turn-1", "prediction": {"is_impossible": None}},
        {"sample_id": "rollout-1-turn-1", "prediction": None},
    ]}
    path = tmp_path / "rec.json"
    path.write_text(json.dumps(rec))
    flags = load_agent_impossible(str(path))
    trig = agent_trigger(_toy(), flags)
    assert trig.tolist() == [False, True, False, False, False]


def test_logistic_gate_and_auc():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    feasible = (X["a"] + 0.3 * rng.normal(size=400)) > 0
    p = logistic_gate(X, feasible).predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert auc(feasible, p) > 0.9
    assert np.isnan(auc(np.ones(5, dtype=bool), np.arange(5)))


def test_overlap_stats_floor_and_two_plus():
    cov = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=bool)
    out = overlap_stats(cov)
    assert out["joint"] == 0.25 and out["miss_two_plus"] == 0.25
    assert np.isclose(out["floor_if_disjoint"], 0.0)  # misses 0.5 + 0.5 + 0
    assert out["ceiling"] == 0.5


def test_summarize_record_direction():
    res = [
        {"prediction": {"remaining_token_interval": [100, 200]}, "ground_truth": {"actual_remaining_total_tokens": 300}},
        {"prediction": {"remaining_token_interval": [100, 200]}, "ground_truth": {"actual_remaining_total_tokens": 50}},
        {"prediction": {"remaining_token_interval": [100, 200]}, "ground_truth": {"actual_remaining_total_tokens": 150}},
        {"prediction": {"remaining_token_interval": None}, "ground_truth": {"actual_remaining_total_tokens": 10}},
    ]
    out = summarize_record(res)
    assert out["n_with_interval"] == 3 and np.isclose(out["no_interval"], 0.25)
    assert np.isclose(out["truth_above (agent too optimistic)"], 1 / 3)
    assert np.isclose(out["truth_below (agent too cautious)"], 1 / 3)
