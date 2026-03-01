"""Panel utilities: bridge between dtk cdh() output and qtk single-series functions."""

from __future__ import annotations

from typing import Callable

import polars as pl

__all__ = ["to_series", "to_series_dict", "to_wide", "apply"]


def to_series(df: pl.DataFrame, ticker: str, field: str) -> pl.DataFrame:
    """Extract one {date, value} series from cdh() output.

    Filters to the given ticker, renames ValueDate->date and field->value.
    """
    return (
        df.filter(pl.col("Ticker") == ticker)
        .select([pl.col("ValueDate").alias("date"), pl.col(field).alias("value")])
        .sort("date")
        .cast({"date": pl.Date, "value": pl.Float64})
    )


def to_series_dict(df: pl.DataFrame, field: str) -> dict[str, pl.DataFrame]:
    """Extract all tickers as {ticker: series} dict from cdh() output."""
    tickers = df["Ticker"].unique().sort().to_list()
    return {t: to_series(df, t, field) for t in tickers}


def to_wide(df: pl.DataFrame, field: str) -> pl.DataFrame:
    """Pivot cdh() output to wide format: date rows, one column per ticker.

    Returns DataFrame with "date" column + one Float64 column per ticker.
    """
    return (
        df.select(["ValueDate", "Ticker", field])
        .rename({"ValueDate": "date"})
        .pivot(on="Ticker", index="date", values=field)
        .sort("date")
    )


def apply(df: pl.DataFrame, field: str, fn: Callable, **kwargs) -> pl.DataFrame:
    """Apply a qtk function to each security's series, return combined result.

    Returns DataFrame with columns: Ticker, date, value.
    Example: panel.apply(data, "PX_CLOSE", qtk.ts.econometrics.returns)
    """
    results = []
    for ticker, series in to_series_dict(df, field).items():
        result = fn(series, **kwargs)
        result = result.with_columns(pl.lit(ticker).alias("Ticker"))
        results.append(result)
    if not results:
        return pl.DataFrame(
            {"Ticker": [], "date": [], "value": []},
            schema={"Ticker": pl.String, "date": pl.Date, "value": pl.Float64},
        )
    return pl.concat(results).select(["Ticker", "date", "value"])
