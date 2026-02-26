"""Tests for ts/downside.py."""

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from qtk.ts.downside import (
    downside_deviation,
    downside_frequency,
    lpm,
    pain_index,
    semi_deviation,
    semi_variance,
    ulcer_index,
    upside_frequency,
    upside_potential_ratio,
    volatility_skewness,
)


def _s(vals, start=dt.date(2024, 1, 1)):
    dates = [start + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"date": dates, "value": [float(v) for v in vals]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def _price_series(n=50, seed=42):
    rng = np.random.default_rng(seed)
    vals = [100.0]
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + rng.standard_normal() * 0.01))
    return pl.DataFrame({"date": dates, "value": vals}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def test_downside_deviation_zero_mar():
    """Returns series all positive => downside deviation should be 0."""
    s = _s([100, 101, 102, 103, 104])
    dd = downside_deviation(s, mar=0.0)
    last = dd["value"].drop_nulls()[-1]
    assert last == pytest.approx(0.0, abs=1e-10)


def test_downside_deviation_positive():
    """With MAR=0 and some negative returns, downside deviation > 0."""
    s = _price_series(50)
    dd = downside_deviation(s, mar=0.0)
    last = dd["value"].drop_nulls()[-1]
    assert last >= 0.0


def test_semi_deviation_returns_series():
    s = _price_series(50)
    result = semi_deviation(s)
    assert result.height > 0
    vals = result["value"].drop_nulls().to_numpy()
    assert all(v >= 0 for v in vals if not np.isnan(v))


def test_semi_variance_equals_squared_semi_deviation():
    """semi_variance = (semi_deviation / 100)^2."""
    s = _price_series(50)
    sd = semi_deviation(s)
    sv = semi_variance(s)
    joined = sd.join(sv, on="date", suffix="_sv").drop_nulls()
    for r in joined.iter_rows():
        sd_val, sv_val = r[1], r[2]
        if not (math.isnan(sd_val) or math.isnan(sv_val)):
            assert sv_val == pytest.approx((sd_val / 100.0) ** 2, rel=1e-6)


def test_lpm_order_1():
    """LPM(1, 0) = mean of max(0 - return, 0) = mean of losses."""
    s = _s([100, 90, 100, 80, 100])
    result = lpm(s, n=1, threshold=0.0)
    last = result["value"].drop_nulls()[-1]
    assert last >= 0.0


def test_lpm_order_2_positive():
    s = _price_series(50)
    result = lpm(s, n=2, threshold=0.0)
    vals = result["value"].drop_nulls().to_numpy()
    assert all(v >= 0 for v in vals)


def test_upside_potential_ratio_positive():
    s = _price_series(50)
    result = upside_potential_ratio(s)
    last = result["value"].drop_nulls()[-1]
    assert last > 0


def test_pain_index_non_negative():
    s = _price_series(50)
    pi = pain_index(s)
    assert pi >= 0.0


def test_ulcer_index_non_negative():
    s = _price_series(50)
    ui = ulcer_index(s)
    assert ui >= 0.0


def test_ulcer_index_flat():
    """Flat series has zero ulcer index."""
    s = _s([100, 100, 100, 100])
    assert ulcer_index(s) == pytest.approx(0.0)


def test_downside_frequency_all_positive():
    """All positive returns => downside frequency = 0."""
    s = _s([100, 101, 102, 103])
    df = downside_frequency(s, mar=0.0)
    last = df["value"].drop_nulls()[-1]
    assert last == pytest.approx(0.0)


def test_downside_upside_frequency_sum_to_one():
    """Downside + upside frequency should sum to at most 1 (ties excluded)."""
    s = _price_series(50)
    dsf = downside_frequency(s).rename({"value": "down"})
    usf = upside_frequency(s).rename({"value": "up"})
    joined = dsf.join(usf, on="date").drop_nulls()
    for r in joined.iter_rows():
        d, u = r[1], r[2]
        assert d + u <= 1.0 + 1e-10


def test_volatility_skewness_shape():
    s = _price_series(100)
    result = volatility_skewness(s)
    assert result.height > 0
