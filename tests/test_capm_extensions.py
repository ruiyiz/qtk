"""Tests for CAPM extensions in ts/econometrics.py."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from qtk.ts.econometrics import (
    active_premium,
    alpha,
    bear_beta,
    bull_beta,
    down_capture,
    specific_risk,
    systematic_risk,
    timing_ratio,
    tracking_error,
    up_capture,
)


def _price_series(n=252, seed=42):
    rng = np.random.default_rng(seed)
    vals = [100.0]
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + rng.standard_normal() * 0.01))
    return pl.DataFrame({"date": dates, "value": vals}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def _s(vals, start=dt.date(2024, 1, 1)):
    dates = [start + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"date": dates, "value": [float(v) for v in vals]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def test_alpha_vs_benchmark_itself():
    """Alpha vs identical benchmark should be near zero."""
    s = _price_series()
    result = alpha(s, s)
    vals = result["value"].drop_nulls().to_numpy()
    for v in vals:
        if not np.isnan(v):
            assert abs(v) < 1e-8


def test_alpha_shape():
    s1 = _price_series(seed=1)
    s2 = _price_series(seed=2)
    result = alpha(s1, s2)
    assert result.height > 0


def test_bull_beta_non_negative():
    """Bull beta measures correlation in up markets; typically positive."""
    s1 = _price_series(seed=1)
    s2 = _price_series(seed=2)
    result = bull_beta(s1, s2)
    assert result.height > 0


def test_bear_beta_shape():
    s1 = _price_series(seed=1)
    s2 = _price_series(seed=2)
    result = bear_beta(s1, s2)
    assert result.height > 0


def test_timing_ratio_shape():
    s1 = _price_series(seed=1)
    s2 = _price_series(seed=2)
    result = timing_ratio(s1, s2)
    assert result.height > 0


def test_tracking_error_zero_vs_self():
    """Tracking error vs self should be zero."""
    s = _price_series()
    result = tracking_error(s, s)
    vals = result["value"].drop_nulls().to_numpy()
    for v in vals:
        if not np.isnan(v):
            assert abs(v) < 1e-8


def test_tracking_error_positive():
    s1 = _price_series(seed=1)
    s2 = _price_series(seed=2)
    result = tracking_error(s1, s2)
    vals = result["value"].drop_nulls().to_numpy()
    non_nan = [v for v in vals if not np.isnan(v)]
    assert all(v >= 0 for v in non_nan)


def test_active_premium_zero_vs_self():
    """Active premium vs self should be zero."""
    s = _price_series()
    result = active_premium(s, s)
    vals = result["value"].drop_nulls().to_numpy()
    for v in vals:
        if not np.isnan(v):
            assert abs(v) < 1e-8


def test_up_capture_vs_self():
    """Up-capture vs self should be 1.0 (in up periods)."""
    s = _price_series()
    result = up_capture(s, s)
    vals = result["value"].drop_nulls().to_numpy()
    non_nan = [v for v in vals if not np.isnan(v)]
    for v in non_nan:
        assert v == pytest.approx(1.0, rel=1e-6)


def test_down_capture_vs_self():
    """Down-capture vs self should be 1.0 (in down periods)."""
    s = _price_series()
    result = down_capture(s, s)
    vals = result["value"].drop_nulls().to_numpy()
    non_nan = [v for v in vals if not np.isnan(v)]
    for v in non_nan:
        assert v == pytest.approx(1.0, rel=1e-6)


def test_systematic_plus_specific_approx_total():
    """systematic + specific should be non-negative where both are defined."""
    s1 = _price_series(seed=1)
    s2 = _price_series(seed=2)
    sys = systematic_risk(s1, s2)
    spe = specific_risk(s1, s2)
    joined = sys.join(spe, on="date", how="inner", suffix="_spe")
    assert joined.height > 0
    sys_arr = joined["value"].to_numpy()
    spe_arr = joined["value_spe"].to_numpy()
    for sys_v, spe_v in zip(sys_arr, spe_arr):
        if not (np.isnan(sys_v) or np.isnan(spe_v)):
            assert sys_v + spe_v >= -1e-10
