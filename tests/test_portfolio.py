"""Tests for qtk.portfolio: Portfolio class."""

import datetime as dt
import math

import polars as pl
import pytest

from qtk.portfolio import Portfolio


def _make_cdh(tickers, n=50, start=dt.date(2024, 1, 2)):
    """Build a mock cdh() DataFrame with realistic price series."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for i, ticker in enumerate(tickers):
        price = 100.0 + i * 50
        for j in range(n):
            price = price * (1 + rng.standard_normal() * 0.01)
            rows.append(
                {
                    "SecurityId": i + 1,
                    "Ticker": ticker,
                    "ValueDate": start + dt.timedelta(days=j),
                    "PX_CLOSE": price,
                }
            )
    return pl.DataFrame(rows).with_columns(
        [
            pl.col("SecurityId").cast(pl.Int32),
            pl.col("ValueDate").cast(pl.Date),
            pl.col("PX_CLOSE").cast(pl.Float64),
        ]
    )


@pytest.fixture
def two_asset_cdh():
    return _make_cdh(["SPY", "AGG"], n=100)


@pytest.fixture
def three_asset_cdh():
    return _make_cdh(["SPY", "AGG", "GLD"], n=100)


@pytest.fixture
def equal_weight_port():
    return Portfolio(weights={"SPY": 0.5, "AGG": 0.5}, name="50/50")


@pytest.fixture
def bench_port():
    return Portfolio(weights={"SPY": 0.6, "AGG": 0.4}, benchmark="SPY", name="60/40")


def test_weights_validation():
    with pytest.raises(ValueError, match="sum to 1.0"):
        Portfolio(weights={"SPY": 0.5, "AGG": 0.3})


def test_returns_columns(equal_weight_port, two_asset_cdh):
    result = equal_weight_port.returns(two_asset_cdh)
    assert result.columns == ["date", "value"]
    assert result.dtypes == [pl.Date, pl.Float64]


def test_returns_length(equal_weight_port, two_asset_cdh):
    result = equal_weight_port.returns(two_asset_cdh)
    # returns drops first observation
    assert len(result) == 99


def test_returns_weighted(two_asset_cdh):
    """Verify weighted return equals weighted average of individual returns."""
    port = Portfolio(weights={"SPY": 0.7, "AGG": 0.3})
    from qtk import panel
    from qtk.ts.econometrics import returns

    spy = panel.to_series(two_asset_cdh, "SPY", "PX_CLOSE")
    agg = panel.to_series(two_asset_cdh, "AGG", "PX_CLOSE")
    spy_ret = returns(spy)
    agg_ret = returns(agg)

    port_ret = port.returns(two_asset_cdh)
    expected = spy_ret.join(agg_ret, on="date", suffix="_agg")
    expected = expected.with_columns(
        (pl.col("value") * 0.7 + pl.col("value_agg") * 0.3).alias("expected")
    )
    joined = port_ret.join(expected.select(["date", "expected"]), on="date")
    for row in joined.iter_rows(named=True):
        assert row["value"] == pytest.approx(row["expected"], rel=1e-6)


def test_cumulative_starts_at_first_return(equal_weight_port, two_asset_cdh):
    cum = equal_weight_port.cumulative(two_asset_cdh)
    assert cum.columns == ["date", "value"]
    # Cumulative series from cum_prod of (1 + ret): first value is 1 + ret[0]
    ret = equal_weight_port.returns(two_asset_cdh)
    expected_first = 1.0 + float(ret["value"][0])
    assert float(cum["value"][0]) == pytest.approx(expected_first, rel=1e-6)


def test_cumulative_monotonically_possible(equal_weight_port, two_asset_cdh):
    cum = equal_weight_port.cumulative(two_asset_cdh)
    assert cum["value"].is_not_nan().all()
    assert len(cum) == 99


def test_contribution_columns(equal_weight_port, two_asset_cdh):
    result = equal_weight_port.contribution(two_asset_cdh)
    assert result.columns == ["Ticker", "date", "value"]


def test_contribution_tickers(equal_weight_port, two_asset_cdh):
    result = equal_weight_port.contribution(two_asset_cdh)
    assert set(result["Ticker"].unique().to_list()) == {"SPY", "AGG"}


def test_contribution_sums_to_portfolio_return(equal_weight_port, two_asset_cdh):
    """Sum of contributions on each date should equal portfolio return."""
    contrib = equal_weight_port.contribution(two_asset_cdh)
    port_ret = equal_weight_port.returns(two_asset_cdh)

    summed = contrib.group_by("date").agg(pl.col("value").sum()).sort("date")
    joined = port_ret.join(summed, on="date", suffix="_sum")
    for row in joined.iter_rows(named=True):
        assert row["value"] == pytest.approx(row["value_sum"], rel=1e-6)


def test_summary_keys(equal_weight_port, two_asset_cdh):
    result = equal_weight_port.summary(two_asset_cdh)
    for key in [
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
    ]:
        assert key in result


def test_summary_max_drawdown_negative(equal_weight_port, two_asset_cdh):
    result = equal_weight_port.summary(two_asset_cdh)
    assert result["max_drawdown"] <= 0.0


def test_summary_volatility_positive(equal_weight_port, two_asset_cdh):
    result = equal_weight_port.summary(two_asset_cdh)
    assert result["volatility"] > 0.0


def test_summary_with_benchmark(bench_port, two_asset_cdh):
    result = bench_port.summary(two_asset_cdh)
    assert "tracking_error" in result
    assert "info_ratio" in result
    assert "beta" in result


def test_summary_benchmark_missing(two_asset_cdh):
    port = Portfolio(weights={"SPY": 0.6, "AGG": 0.4}, benchmark="GLD")
    result = port.summary(two_asset_cdh)
    # GLD not in price_data, so benchmark metrics should be absent
    assert "tracking_error" not in result
    assert "info_ratio" not in result


def test_equal_weight_total_return_finite(equal_weight_port, two_asset_cdh):
    result = equal_weight_port.summary(two_asset_cdh)
    assert math.isfinite(result["total_return"])
    assert math.isfinite(result["annualized_return"])
