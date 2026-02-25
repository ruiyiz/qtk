import datetime as dt

import polars as pl
import pytest

from qtk.ts.technicals import (
    SeasonalModel,
    Frequency,
    bollinger_bands,
    exponential_moving_average,
    exponential_spread_volatility,
    exponential_volatility,
    macd,
    moving_average,
    relative_strength_index,
    smoothed_moving_average,
)


def _s(vals, start=dt.date(2024, 1, 1)):
    dates = [start + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"date": dates, "value": [float(v) for v in vals]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def _price_series(n=60):
    import numpy as np
    rng = np.random.default_rng(42)
    vals = [100.0]
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    for _ in range(n - 1):
        vals.append(vals[-1] + rng.standard_normal())
    return pl.DataFrame({"date": dates, "value": vals}).cast({"date": pl.Date, "value": pl.Float64})


def test_moving_average_full():
    s = _s([1.0, 2.0, 3.0, 4.0, 5.0])
    result = moving_average(s, 3)
    assert result["value"][-1] == pytest.approx(4.0)


def test_bollinger_bands_schema():
    s = _price_series(30)
    result = bollinger_bands(s, 10)
    assert "lower" in result.columns
    assert "upper" in result.columns
    # upper > lower always
    joined = result.drop_nulls()
    assert all(u > l for u, l in zip(joined["upper"].to_list(), joined["lower"].to_list()))


def test_bollinger_bands_centered():
    s = _price_series(30)
    result = bollinger_bands(s, 10, k=2)
    ma = moving_average(s, 10)
    joined = result.join(ma, on="date", how="inner")
    # mean of upper + lower = 2 * ma (approximately)
    midpoint = (joined["upper"] + joined["lower"]) / 2
    assert all(abs(m - v) < 1e-6 for m, v in zip(midpoint.to_list(), joined["value"].to_list()))


def test_smoothed_moving_average():
    s = _price_series(20)
    result = smoothed_moving_average(s, 5)
    # normalize_window(s, 5) → Window(5, 5); ramp drops first 5 rows
    assert result.height == s.height - 5
    assert all(v > 0 for v in result["value"].to_list())


def test_rsi_range():
    s = _price_series(50)
    result = relative_strength_index(s, 14)
    non_null = result.drop_nulls("value")
    assert all(0 <= v <= 100 for v in non_null["value"].to_list())


def test_exponential_moving_average():
    s = _s([1.0, 2.0, 3.0, 4.0, 5.0])
    result = exponential_moving_average(s, beta=0.5)
    assert result.height == 5
    # First value should equal first input value
    assert result["value"][0] == pytest.approx(1.0)


def test_macd():
    s = _price_series(50)
    result = macd(s, m=5, n=10, s=2)
    assert result.height == s.height


def test_exponential_volatility():
    s = _price_series(50)
    result = exponential_volatility(s, beta=0.9)
    non_null = result.drop_nulls("value")
    # EWM std starts at 0.0 for the first obs; remaining values should be >= 0
    assert all(v >= 0 for v in non_null["value"].to_list())


def test_exponential_spread_volatility():
    s = _price_series(50)
    result = exponential_spread_volatility(s, beta=0.9)
    non_null = result.drop_nulls("value")
    # EWM std starts at 0.0 for the first obs; remaining values should be >= 0
    assert all(v >= 0 for v in non_null["value"].to_list())
