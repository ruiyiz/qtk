"""Tests for qtk.position: PositionHistory class."""

import datetime as dt
import math

import polars as pl
import pytest

from qtk.portfolio import Portfolio
from qtk.position import PositionHistory


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


def _make_holdings(tickers, as_of, shares_list):
    rows = [
        {"date": as_of, "ticker": t, "shares": float(s)}
        for t, s in zip(tickers, shares_list)
    ]
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "ticker": pl.String, "shares": pl.Float64},
    )


@pytest.fixture
def two_asset_cdh():
    return _make_cdh(["SPY", "AGG"], n=100)


@pytest.fixture
def equal_shares_ph(two_asset_cdh):
    d0 = two_asset_cdh["ValueDate"].min()
    holdings = _make_holdings(["SPY", "AGG"], d0, [100.0, 100.0])
    return PositionHistory(holdings=holdings)


def test_position_history_validation():
    bad = pl.DataFrame({"date": [], "ticker": []})
    with pytest.raises(ValueError, match="missing required columns"):
        PositionHistory(holdings=bad)


def test_fill_holdings_forward_fill():
    d1 = dt.date(2024, 1, 1)
    holdings = pl.DataFrame(
        {"date": [d1], "ticker": ["SPY"], "shares": [100.0]},
        schema={"date": pl.Date, "ticker": pl.String, "shares": pl.Float64},
    )
    ph = PositionHistory(holdings=holdings)
    price_data = _make_cdh(["SPY"], n=5, start=d1)
    filled = ph._fill_holdings(price_data)
    assert len(filled) == 5
    assert (filled["shares"] == 100.0).all()


def test_weights_sum_to_one(equal_shares_ph, two_asset_cdh):
    w = equal_shares_ph.weights(two_asset_cdh)
    sums = w.group_by("date").agg(pl.col("weight").sum().alias("s"))
    for val in sums["s"].to_list():
        assert abs(val - 1.0) < 1e-9


def test_returns_columns_and_dtype(equal_shares_ph, two_asset_cdh):
    ret = equal_shares_ph.returns(two_asset_cdh)
    assert ret.columns == ["date", "value"]
    assert ret.dtypes == [pl.Date, pl.Float64]


def test_returns_static_matches_portfolio():
    """For a single return period, PositionHistory with matched initial shares
    produces the same return as a static Portfolio."""
    d1 = dt.date(2024, 1, 1)
    d2 = dt.date(2024, 1, 2)

    price_data = pl.DataFrame(
        {
            "SecurityId": [1, 1, 2, 2],
            "Ticker": ["SPY", "SPY", "AGG", "AGG"],
            "ValueDate": [d1, d2, d1, d2],
            "PX_CLOSE": [100.0, 102.0, 50.0, 51.0],
        }
    ).with_columns(
        [pl.col("ValueDate").cast(pl.Date), pl.col("PX_CLOSE").cast(pl.Float64)]
    )

    # Set shares so initial weights are exactly 0.5/0.5:
    # shares_spy * 100 = shares_agg * 50 = 0.5 * total_mv
    # total_mv = 1.0 => shares_spy = 0.5/100, shares_agg = 0.5/50
    spy_shares = 0.5 / 100.0
    agg_shares = 0.5 / 50.0

    holdings = pl.DataFrame(
        {
            "date": [d1, d1],
            "ticker": ["SPY", "AGG"],
            "shares": [spy_shares, agg_shares],
        },
        schema={"date": pl.Date, "ticker": pl.String, "shares": pl.Float64},
    )
    ph = PositionHistory(holdings=holdings)
    ph_ret = ph.returns(price_data)

    port = Portfolio(weights={"SPY": 0.5, "AGG": 0.5})
    port_ret = port.returns(price_data)

    assert len(ph_ret) == 1
    assert len(port_ret) == 1
    assert float(ph_ret["value"][0]) == pytest.approx(
        float(port_ret["value"][0]), rel=1e-6
    )


def test_returns_with_rebalance():
    """After position change, new prior-period weights are reflected in returns."""
    d1 = dt.date(2024, 1, 1)
    d2 = dt.date(2024, 1, 2)
    d3 = dt.date(2024, 1, 3)

    price_data = pl.DataFrame(
        {
            "SecurityId": [1, 1, 1, 2, 2, 2],
            "Ticker": ["SPY"] * 3 + ["AGG"] * 3,
            "ValueDate": [d1, d2, d3, d1, d2, d3],
            "PX_CLOSE": [100.0, 102.0, 101.0, 50.0, 51.0, 52.0],
        }
    ).with_columns(
        [pl.col("ValueDate").cast(pl.Date), pl.col("PX_CLOSE").cast(pl.Float64)]
    )

    # Holdings: SPY=100 at d1; at d2 rebalance to SPY=50, AGG=100
    holdings = pl.DataFrame(
        {
            "date": [d1, d2, d2],
            "ticker": ["SPY", "SPY", "AGG"],
            "shares": [100.0, 50.0, 100.0],
        },
        schema={"date": pl.Date, "ticker": pl.String, "shares": pl.Float64},
    )
    ph = PositionHistory(holdings=holdings)
    ret = ph.returns(price_data)

    assert len(ret) == 2

    # At d2: prior weights (d1) = SPY=100%, AGG=0%
    # SPY return d1->d2 = 102/100 - 1 = 0.02
    r_d2 = float(ret.filter(pl.col("date") == d2)["value"][0])
    assert r_d2 == pytest.approx(0.02, rel=1e-6)

    # At d3: prior weights (d2) = SPY: 50*102 / (50*102 + 100*51) = 5100/10200 = 0.5
    #                             AGG: 100*51 / (50*102 + 100*51) = 5100/10200 = 0.5
    spy_ret_d3 = 101.0 / 102.0 - 1
    agg_ret_d3 = 52.0 / 51.0 - 1
    spy_mv_d2 = 50.0 * 102.0
    agg_mv_d2 = 100.0 * 51.0
    total_d2 = spy_mv_d2 + agg_mv_d2
    expected_d3 = (spy_mv_d2 / total_d2) * spy_ret_d3 + (
        agg_mv_d2 / total_d2
    ) * agg_ret_d3
    r_d3 = float(ret.filter(pl.col("date") == d3)["value"][0])
    assert r_d3 == pytest.approx(expected_d3, rel=1e-6)


def test_cumulative(equal_shares_ph, two_asset_cdh):
    cum = equal_shares_ph.cumulative(two_asset_cdh)
    assert cum.columns == ["date", "value"]
    assert len(cum) > 0
    # First cumulative value = 1 + first return
    ret = equal_shares_ph.returns(two_asset_cdh)
    expected_first = 1.0 + float(ret["value"][0])
    assert float(cum["value"][0]) == pytest.approx(expected_first, rel=1e-6)


def test_contribution_sums_to_return(equal_shares_ph, two_asset_cdh):
    contrib = equal_shares_ph.contribution(two_asset_cdh)
    port_ret = equal_shares_ph.returns(two_asset_cdh)

    assert set(contrib.columns) == {"Ticker", "date", "value"}

    summed = contrib.group_by("date").agg(pl.col("value").sum()).sort("date")
    joined = port_ret.join(summed, on="date", suffix="_sum")
    for row in joined.iter_rows(named=True):
        assert row["value"] == pytest.approx(row["value_sum"], rel=1e-6)


def test_summary_keys(equal_shares_ph, two_asset_cdh):
    result = equal_shares_ph.summary(two_asset_cdh)
    for key in [
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
    ]:
        assert key in result


def test_summary_max_drawdown_non_positive(equal_shares_ph, two_asset_cdh):
    result = equal_shares_ph.summary(two_asset_cdh)
    assert result["max_drawdown"] <= 0.0


def test_summary_volatility_positive(equal_shares_ph, two_asset_cdh):
    result = equal_shares_ph.summary(two_asset_cdh)
    assert result["volatility"] > 0.0


def test_summary_total_return_finite(equal_shares_ph, two_asset_cdh):
    result = equal_shares_ph.summary(two_asset_cdh)
    assert math.isfinite(result["total_return"])
    assert math.isfinite(result["annualized_return"])


def test_from_weights_factory():
    weights = {"SPY": 0.6, "AGG": 0.4}
    ph = PositionHistory.from_weights(weights, as_of=dt.date(2024, 1, 2))
    assert set(ph.holdings.columns) == {"date", "ticker", "shares"}
    assert len(ph.holdings) == 2
    tickers = set(ph.holdings["ticker"].to_list())
    assert tickers == {"SPY", "AGG"}
    shares_map = dict(
        zip(ph.holdings["ticker"].to_list(), ph.holdings["shares"].to_list())
    )
    assert shares_map["SPY"] == pytest.approx(0.6)
    assert shares_map["AGG"] == pytest.approx(0.4)


def test_to_portfolio_snapshot(equal_shares_ph, two_asset_cdh):
    dt_snap = two_asset_cdh["ValueDate"].min()
    port = equal_shares_ph.to_portfolio(two_asset_cdh, dt_snap)
    assert isinstance(port, Portfolio)
    assert set(port.weights.keys()) == {"SPY", "AGG"}
    total_w = sum(port.weights.values())
    assert total_w == pytest.approx(1.0, rel=1e-6)
