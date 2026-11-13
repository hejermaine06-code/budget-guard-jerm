"""Tests for the resource-specific ('which-budget-was-hit') censored calibration."""
import numpy as np

from budget_guard.censored import gated_upper_qhat, km_upper_qhat, upper_qhat


def test_resource_bound_threads_between_drop_and_bound():
    """drop ignores censoring (smallest qhat); bound applies every censored bound
    incl. false cross-resource ones (largest); resource_bound sits between by keeping
    only the operative-resource bounds."""
    rng = np.random.default_rng(0)
    s_unc = rng.normal(0.0, 1.0, size=600)
    binding = np.array([True, False] * 300)
    # binding-resource bounds are moderate; non-binding (other resource) bounds are
    # large and false-for-this-resource — exactly what inflates the naive `bound`.
    lb = np.where(binding, rng.uniform(2, 4, 600), rng.uniform(40, 80, 600))

    q_drop = upper_qhat(s_unc, lb, 0.1, "drop")
    q_res = upper_qhat(s_unc, lb, 0.1, "resource_bound", censored_binding=binding)
    q_bound = upper_qhat(s_unc, lb, 0.1, "bound")

    assert q_drop < q_res < q_bound


def test_resource_bound_equals_bound_on_the_binding_subset():
    rng = np.random.default_rng(1)
    s_unc = rng.normal(size=200)
    binding = rng.random(120) < 0.5
    lb = rng.uniform(1, 5, 120)

    q_res = upper_qhat(s_unc, lb, 0.1, "resource_bound", censored_binding=binding)
    q_manual = upper_qhat(s_unc, lb[binding], 0.1, "bound")
    assert np.isclose(q_res, q_manual)


def test_resource_bound_without_mask_falls_back_to_bound():
    rng = np.random.default_rng(2)
    s_unc = rng.normal(size=200)
    lb = rng.uniform(1, 5, 80)
    assert np.isclose(
        upper_qhat(s_unc, lb, 0.1, "resource_bound"),
        upper_qhat(s_unc, lb, 0.1, "bound"),
    )


def test_gated_upper_qhat_hits_nominal_on_kept_scores():
    """On the kept (gated) uncensored scores, split conformal must hit ~1-alpha."""
    rng = np.random.default_rng(0)
    s = rng.normal(size=2000)
    q = gated_upper_qhat(s, 0.1)
    assert 0.86 <= float(np.mean(s <= q)) <= 0.94


def test_gated_upper_qhat_empty_is_inf():
    assert gated_upper_qhat(np.array([]), 0.1) == float("inf")


def test_km_matches_plain_quantile_without_censoring():
    rng = np.random.default_rng(0)
    s = rng.normal(size=1000)
    q = km_upper_qhat(s, np.array([]), 0.1)
    assert abs(q - float(np.quantile(s, 0.9, method="higher"))) < 0.2


def test_km_diverges_when_upper_tail_is_censored():
    """Events all small, a large mass of high right-censored obs -> S(t) never <= alpha."""
    s_events = np.linspace(0.0, 1.0, 50)
    s_censored = np.full(200, 100.0)
    assert km_upper_qhat(s_events, s_censored, 0.1) == float("inf")


def test_corrected_quantile_is_infinite_on_tiny_samples():
    """Code review #8: too few scores must give +inf, not the largest score."""
    from budget_guard.censored import _corrected_quantile, gated_upper_qhat, upper_qhat

    assert _corrected_quantile(np.array([1.0, 2.0, 3.0]), 0.1) == np.inf
    assert _corrected_quantile(np.array([]), 0.1) == np.inf
    assert gated_upper_qhat(np.arange(5, dtype=float), 0.1) == np.inf
    assert np.isfinite(_corrected_quantile(np.arange(20, dtype=float), 0.1))
    assert upper_qhat(np.array([]), np.array([]), 0.1, "drop") == np.inf
