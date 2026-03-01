"""Tests for qtk.panel: multi-asset bridge layer."""

import datetime as dt

import polars as pl
import pytest

from qtk import panel
from qtk.ts.econometrics import returns


def _make_cdh(tickers, n=10, start=dt.date(2024, 1, 2)):
    """Build a mock cdh() DataFrame for testing."""
    rows = []
    for i, ticker in enumerate(tickers):
        base = 100.0 + i * 50
        for j in range(n):
            rows.append(
                {
                    "SecurityId": i + 1,
                    "Ticker": ticker,
                    "ValueDate": start + dt.timedelta(days=j),
                    "PX_CLOSE": base + j,
                    "PX_LAST": base + j + 0.1,
                }
            )
    return pl.DataFrame(rows).with_columns(
        [
            pl.col("SecurityId").cast(pl.Int32),
            pl.col("ValueDate").cast(pl.Date),
            pl.col("PX_CLOSE").cast(pl.Float64),
            pl.col("PX_LAST").cast(pl.Float64),
        ]
    )


@pytest.fixture
def cdh_df():
    return _make_cdh(["SPY", "AAPL"], n=10)


def test_to_series_columns(cdh_df):
    result = panel.to_series(cdh_df, "SPY", "PX_CLOSE")
    assert result.columns == ["date", "value"]
    assert result.dtypes == [pl.Date, pl.Float64]


def test_to_series_length(cdh_df):
    result = panel.to_series(cdh_df, "SPY", "PX_CLOSE")
    assert len(result) == 10


def test_to_series_values(cdh_df):
    result = panel.to_series(cdh_df, "SPY", "PX_CLOSE")
    assert result["value"][0] == pytest.approx(100.0)


def test_to_series_sorted(cdh_df):
    result = panel.to_series(cdh_df, "AAPL", "PX_CLOSE")
    dates = result["date"].to_list()
    assert dates == sorted(dates)


def test_to_series_dict_keys(cdh_df):
    result = panel.to_series_dict(cdh_df, "PX_CLOSE")
    assert set(result.keys()) == {"SPY", "AAPL"}


def test_to_series_dict_values(cdh_df):
    result = panel.to_series_dict(cdh_df, "PX_CLOSE")
    for series in result.values():
        assert series.columns == ["date", "value"]
        assert len(series) == 10


def test_to_wide_shape(cdh_df):
    result = panel.to_wide(cdh_df, "PX_CLOSE")
    assert "date" in result.columns
    assert "SPY" in result.columns
    assert "AAPL" in result.columns
    assert len(result) == 10


def test_to_wide_sorted(cdh_df):
    result = panel.to_wide(cdh_df, "PX_CLOSE")
    dates = result["date"].to_list()
    assert dates == sorted(dates)


def test_to_wide_values(cdh_df):
    result = panel.to_wide(cdh_df, "PX_CLOSE")
    assert result["SPY"][0] == pytest.approx(100.0)
    assert result["AAPL"][0] == pytest.approx(150.0)


def test_apply_returns_all_tickers(cdh_df):
    result = panel.apply(cdh_df, "PX_CLOSE", returns)
    assert "Ticker" in result.columns
    assert set(result["Ticker"].unique().to_list()) == {"SPY", "AAPL"}


def test_apply_columns(cdh_df):
    result = panel.apply(cdh_df, "PX_CLOSE", returns)
    assert result.columns == ["Ticker", "date", "value"]


def test_apply_return_count(cdh_df):
    result = panel.apply(cdh_df, "PX_CLOSE", returns)
    # returns drops first obs per ticker: 2 tickers * (10-1) = 18
    assert len(result) == 18


def test_apply_empty():
    empty_df = pl.DataFrame(
        {"SecurityId": [], "Ticker": [], "ValueDate": [], "PX_CLOSE": []},
        schema={
            "SecurityId": pl.Int32,
            "Ticker": pl.String,
            "ValueDate": pl.Date,
            "PX_CLOSE": pl.Float64,
        },
    )
    result = panel.apply(empty_df, "PX_CLOSE", returns)
    assert result.columns == ["Ticker", "date", "value"]
    assert len(result) == 0


def test_single_ticker_cdh():
    cdh = _make_cdh(["SPY"], n=5)
    result = panel.to_wide(cdh, "PX_CLOSE")
    assert result.columns == ["date", "SPY"]
    assert len(result) == 5
