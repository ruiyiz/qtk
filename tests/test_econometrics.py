import datetime as dt
import math

import polars as pl
import pytest

from qtk.ts.econometrics import (
    AnnualizationFactor,
    annualize,
    beta,
    change,
    correlation,
    excess_returns,
    excess_returns_pure,
    index,
    max_drawdown,
    prices,
    returns,
    volatility,
    vol_swap_volatility,
)
from qtk.ts.helper import Returns, Window


def _s(vals, start=dt.date(2024, 1, 1)):
    dates = [start + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"date": dates, "value": [float(v) for v in vals]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def _price_series(n=50):
    import numpy as np
    rng = np.random.default_rng(99)
    vals = [100.0]
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + rng.standard_normal() * 0.01))
    return pl.DataFrame({"date": dates, "value": vals}).cast({"date": pl.Date, "value": pl.Float64})


def test_returns_simple():
    s = _s([100.0, 110.0, 121.0])
    result = returns(s)
    assert abs(result["value"][0] - 0.1) < 1e-10
    assert abs(result["value"][1] - 0.1) < 1e-10


def test_returns_log():
    s = _s([100.0, math.e * 100])
    result = returns(s, type=Returns.LOGARITHMIC)
    assert abs(result["value"][0] - 1.0) < 1e-10


def test_returns_absolute():
    s = _s([100.0, 105.0])
    result = returns(s, type=Returns.ABSOLUTE)
    assert result["value"][0] == pytest.approx(5.0)


def test_prices_from_returns():
    s = _s([100.0, 110.0, 121.0])
    ret = returns(s)
    rec = prices(ret, initial=100.0)
    # prices(returns(prices)) should give back roughly original after first obs
    assert abs(rec["value"][-1] - 121.0) < 0.001


def test_index_normalize():
    s = _s([100.0, 200.0, 300.0])
    result = index(s, initial=1.0)
    assert result["value"][0] == pytest.approx(1.0)
    assert result["value"][-1] == pytest.approx(3.0)


def test_change():
    s = _s([10.0, 15.0, 8.0])
    result = change(s)
    assert result["value"][0] == 0.0
    assert result["value"][1] == pytest.approx(5.0)
    assert result["value"][2] == pytest.approx(-2.0)


def test_annualize_daily():
    s = _price_series(30)
    ret = returns(s)
    result = annualize(ret)
    # Should scale by sqrt(252)
    factor = result["value"][0] / ret["value"][0]
    assert abs(factor - math.sqrt(252)) < 1e-6


def test_volatility_not_empty(price_series):
    result = volatility(price_series, 22)
    assert not result.is_empty()
    assert all(v > 0 for v in result.drop_nulls("value")["value"].to_list())


def test_vol_swap_volatility(price_series):
    result = vol_swap_volatility(price_series, n_days=22)
    assert not result.is_empty()


def test_correlation_self(price_series):
    result = correlation(price_series, price_series, 22)
    non_null = result.drop_nulls("value")
    assert all(abs(v - 1.0) < 0.01 for v in non_null["value"].to_list())


def test_beta_self(price_series):
    result = beta(price_series, price_series, 22)
    non_null = result.drop_nulls("value")
    assert all(abs(v - 1.0) < 0.01 for v in non_null["value"].to_list())


def test_max_drawdown_negative(price_series):
    result = max_drawdown(price_series)
    non_null = result.drop_nulls("value")
    assert all(v <= 0 for v in non_null["value"].to_list())


def test_excess_returns_float():
    s = _s([100.0, 101.0, 102.0])
    result = excess_returns(s, 0.05)
    assert result.height == 3
    assert result["value"][0] == pytest.approx(100.0)


def test_excess_returns_pure():
    prices_df = _s([100.0, 102.0, 101.0])
    spot = _s([1.0, 1.01, 1.02])
    result = excess_returns_pure(prices_df, spot)
    assert result.height == 3
    assert result["value"][0] == pytest.approx(100.0)


def test_excess_returns_non_float_raises():
    with pytest.raises(Exception):
        excess_returns(_s([100.0, 101.0]), "USD")
