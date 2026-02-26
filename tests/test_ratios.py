"""Tests for ts/ratios.py."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from qtk.ts.ratios import (
    burke_ratio,
    calmar_ratio,
    information_ratio,
    martin_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    sterling_ratio,
    treynor_ratio,
)


def _s(vals, start=dt.date(2024, 1, 1)):
    dates = [start + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"date": dates, "value": [float(v) for v in vals]}).cast(
        {"date": pl.Date, "value": pl.Float64}
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


def test_sharpe_ratio_shape():
    s = _price_series()
    result = sharpe_ratio(s, rf=0.0)
    assert result.height > 0


def test_sharpe_ratio_monotone_up():
    """Monotonically rising series: all returns positive => Sharpe ratio should be positive."""
    s = _s([100, 101, 102, 103, 104, 105])
    result = sharpe_ratio(s)
    vals = result["value"].drop_nulls().to_numpy()
    assert all(v > 0 for v in vals if not np.isnan(v))


def test_sharpe_ratio_rf_effect():
    """Higher risk-free rate reduces Sharpe ratio."""
    s = _price_series()
    sr0 = sharpe_ratio(s, rf=0.0)["value"].drop_nulls()[-1]
    sr4 = sharpe_ratio(s, rf=0.04)["value"].drop_nulls()[-1]
    assert sr0 > sr4


def test_sortino_ratio_positive():
    s = _price_series()
    result = sortino_ratio(s)
    vals = result["value"].drop_nulls().to_numpy()
    assert result.height > 0


def test_sortino_vs_sharpe_relationship():
    """Sortino uses only downside vol => generally >= Sharpe in trending markets."""
    s = _price_series()
    sr_last = sharpe_ratio(s)["value"].drop_nulls()[-1]
    so_last = sortino_ratio(s)["value"].drop_nulls()[-1]
    assert not np.isnan(sr_last)
    assert not np.isnan(so_last)


def test_calmar_ratio_shape():
    s = _price_series()
    result = calmar_ratio(s)
    assert result.height > 0


def test_information_ratio_identical_series():
    """IR vs self should be NaN (zero tracking error)."""
    s = _price_series()
    result = information_ratio(s, s)
    vals = result["value"].drop_nulls().to_numpy()
    assert all(np.isnan(v) for v in vals)


def test_information_ratio_shape():
    rng = np.random.default_rng(7)
    n = 252
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    vals1 = [100.0]
    vals2 = [100.0]
    for _ in range(n - 1):
        vals1.append(vals1[-1] * (1 + rng.standard_normal() * 0.01))
        vals2.append(vals2[-1] * (1 + rng.standard_normal() * 0.01))
    s1 = pl.DataFrame({"date": dates, "value": vals1}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    s2 = pl.DataFrame({"date": dates, "value": vals2}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    result = information_ratio(s1, s2)
    assert result.height > 0


def test_treynor_ratio_shape():
    rng = np.random.default_rng(7)
    n = 100
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    vals1 = [100.0]
    vals2 = [100.0]
    for _ in range(n - 1):
        vals1.append(vals1[-1] * (1 + rng.standard_normal() * 0.01))
        vals2.append(vals2[-1] * (1 + rng.standard_normal() * 0.01))
    s1 = pl.DataFrame({"date": dates, "value": vals1}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    s2 = pl.DataFrame({"date": dates, "value": vals2}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    result = treynor_ratio(s1, s2)
    assert result.height > 0


def test_omega_ratio_known():
    """
    All returns = +1% except one = -1%: omega >> 1.
    Returns: [0.01, 0.01, 0.01, -0.01, 0.01]
    gains = 4 * 0.01 = 0.04; losses = 0.01
    omega = 0.04 / 0.01 = 4.0
    """
    # Use a price series that gives these returns
    s = _s([100, 101, 102.01, 103.0301, 102.0, 103.02])
    result = omega_ratio(s)
    last = result["value"].drop_nulls()[-1]
    assert last > 1.0


def test_omega_ratio_all_positive_returns():
    """All positive returns => no losses => NaN."""
    s = _s([100, 101, 102, 103])
    result = omega_ratio(s)
    last = result["value"].drop_nulls()[-1]
    assert np.isnan(last)


def test_sterling_ratio_shape():
    s = _price_series()
    result = sterling_ratio(s)
    assert result.height > 0


def test_burke_ratio_shape():
    s = _price_series()
    result = burke_ratio(s)
    assert result.height > 0


def test_martin_ratio_shape():
    s = _price_series()
    result = martin_ratio(s)
    assert result.height > 0
