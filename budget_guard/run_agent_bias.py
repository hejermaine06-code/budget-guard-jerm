"""budget_guard/run_agent_bias.py — HOW the agents' own budget estimates miss.

The paper reports that agents' self-reported intervals cover the truth only 2-52% of
the time. For the IAB workshop (interpreting agent behavior) the direction matters:
does an agent think it needs LESS than it does (over-optimistic: truth above its
interval) or MORE (over-cautious: truth below it)? BAGEN reports over-optimism; the
Figure 1 run shows the opposite. This counts both, per model x environment, straight
from the BAGEN estimation records (prediction interval vs ground-truth remaining
tokens) — no model fitting, no splits.

Nothing existing is changed. Writes tables/agent_bias.csv.
Usage: python -m budget_guard.run_agent_bias
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "bagen_hf", "estimation")


def summarize_record(results: list[dict]) -> dict:
    """Shares of samples where the truth sits below / inside / above the agent's
    interval, among samples that HAVE an interval, plus the share with none."""
    below = inside = above = missing = 0
    ratios = []
    for r in results:
        iv = (r.get("prediction") or {}).get("remaining_token_interval")
        truth = (r.get("ground_truth") or {}).get("actual_remaining_total_tokens")
        if truth is None:
            continue
        if not (isinstance(iv, (list, tuple)) and len(iv) == 2 and None not in iv):
            missing += 1
            continue
        lo, hi = float(iv[0]), float(iv[1])
        t = float(truth)
        if t < lo:
            below += 1
        elif t > hi:
            above += 1
        else:
            inside += 1
        if t > 0:
            ratios.append((lo + hi) / 2 / t)
    n = below + inside + above
    total = n + missing
    return {
        "n_with_interval": n,
        "inside": inside / n if n else np.nan,
        "truth_above (agent too optimistic)": above / n if n else np.nan,
        "truth_below (agent too cautious)": below / n if n else np.nan,
        "no_interval": missing / total if total else np.nan,
        "median_midpoint_over_truth": float(np.median(ratios)) if ratios else np.nan,
    }


def main() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(f"{_DATA}/*/*.json")):
        run_dir = os.path.basename(os.path.dirname(path))
        if "dialogues" in path or path.endswith("_pairs.json") or run_dir.startswith("newwarehouse"):
            continue
        with open(path, encoding="utf-8") as handle:
            results = json.load(handle).get("results") or []
        env = run_dir.split("-", 1)[0]
        model = run_dir.split("-origin-", 1)[-1].split("_")[0].replace("-low-thinking", "")
        model = model.replace("-128-main", "")
        rows.append({"env": env, "model": model[:28], **summarize_record(results)})
    table = pd.DataFrame(rows).sort_values(["env", "model"]).reset_index(drop=True)
    os.makedirs("tables", exist_ok=True)
    table.to_csv("tables/agent_bias.csv", index=False)
    with pd.option_context("display.width", 220):
        print(table.round(3).to_string(index=False))
    print("\nwrote tables/agent_bias.csv")
    return table


if __name__ == "__main__":
    main()
