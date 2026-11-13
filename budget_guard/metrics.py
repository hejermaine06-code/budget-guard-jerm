"""budget_guard/metrics.py — interval quality metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def marginal_coverage(y, lo, hi) -> float:
    """Fraction of samples whose interval contains the truth."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((lo <= y) & (y <= hi)))


def rollout_coverage(y, lo, hi, groups) -> float:
    """Fraction of rollouts whose interval contains y at EVERY prefix.

    The object ``conformal.cqr_split_rollout`` guarantees; reported for every method
    so pooled (per-prefix) and rollout-level calibration can be compared fairly.
    """
    covered = (np.asarray(lo) <= np.asarray(y)) & (np.asarray(y) <= np.asarray(hi))
    groups = np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    all_covered = np.ones(inverse.max() + 1, dtype=bool)
    np.logical_and.at(all_covered, inverse, covered)
    return float(all_covered.mean())


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def median_width(lo, hi) -> float:
    """Robust to the occasional unbounded interval proper ACI can emit."""
    return float(np.median(np.asarray(hi) - np.asarray(lo)))


def frac_unbounded(hi) -> float:
    return float(np.mean(~np.isfinite(np.asarray(hi))))


def rollout_bootstrap_mean_draws(values, groups, B: int = 500, seed: int = 0):
    """The B rollout-resampled mean statistics behind ``rollout_bootstrap_mean_ci``.

    Exposed so multi-seed drivers can POOL draws across seeds before taking
    quantiles. Averaging per-seed CI *endpoints* instead — what the drivers did
    before 2026-08-01 — is not a confidence interval at any level: seeds whose
    coverage is 1.0 contribute degenerate [1, 1] intervals whose mean swallows the
    seeds that landed at 0.70, understating the plotted band by up to an order of
    magnitude. Pooling keeps both variance sources (within-seed rollout resampling
    and between-seed calibration-split spread) in one honest interval.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if len(values) != len(groups):
        raise ValueError("values and groups must have equal length")
    if len(values) == 0:
        return np.full(B, np.nan)
    uniq, inv = np.unique(groups, return_inverse=True)
    value_sum = np.bincount(inv, weights=values)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(B, len(uniq)))
    return value_sum[idx].sum(axis=1) / counts[idx].sum(axis=1)


def rollout_bootstrap_mean_ci(
    values, groups, B: int = 500, level: float = 0.9, seed: int = 0
):
    """Mean CI that resamples whole rollouts, preserving correlated prefix rows."""
    stats = rollout_bootstrap_mean_draws(values, groups, B=B, seed=seed)
    if not np.isfinite(stats).any():
        return float("nan"), float("nan")
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))


def rollout_bootstrap_ci(y, lo, hi, groups, B: int = 500, level: float = 0.9, seed: int = 0):
    """Coverage CI that respects within-rollout correlation: resample ROLLOUTS with
    replacement, not prefix samples. Per-sample binomial CIs are too tight here."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    covered = ((lo <= y) & (y <= hi)).astype(float)
    return rollout_bootstrap_mean_ci(covered, groups, B=B, level=level, seed=seed)


def rollout_bootstrap_draws(y, lo, hi, groups, B: int = 500, seed: int = 0):
    """Coverage bootstrap draws (see ``rollout_bootstrap_mean_draws``) for pooling."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    covered = ((lo <= y) & (y <= hi)).astype(float)
    return rollout_bootstrap_mean_draws(covered, groups, B=B, seed=seed)


def pooled_bootstrap_ci(draws_per_seed, level: float = 0.9):
    """One interval from per-seed bootstrap draws: pool, then take quantiles.

    Each seed contributes equally many draws, so the pooled distribution is the
    bootstrap distribution of coverage under (seed uniform, rollouts resampled) —
    the cross-seed aggregate the tables report. This replaces mean-of-endpoints
    aggregation; see ``rollout_bootstrap_mean_draws`` for why that was wrong.
    """
    arrays = [np.asarray(d, dtype=float) for d in draws_per_seed]
    if not arrays:
        return float("nan"), float("nan")
    pooled = np.concatenate(arrays)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return float("nan"), float("nan")
    tail = (1 - level) / 2
    return float(np.quantile(pooled, tail)), float(np.quantile(pooled, 1 - tail))


def normalized_width(y, lo, hi, agg: str = "mean") -> float:
    """Width relative to y's central 80% range — comparable across envs whose
    token scales differ by orders of magnitude. agg='median' is robust to
    unbounded intervals."""
    y = np.asarray(y)
    spread = float(np.quantile(y, 0.9) - np.quantile(y, 0.1))
    if spread <= 0:
        return float("nan")
    width = median_width(lo, hi) if agg == "median" else mean_width(lo, hi)
    return width / spread
