"""Shared fixtures for qtk timeseries tests."""

import datetime as dt

import polars as pl
import pytest


@pytest.fixture
def price_series():
    """252 daily observations starting 2023-01-02 with known first value 100."""
    start = dt.date(2023, 1, 2)
    dates = [start + dt.timedelta(days=i) for i in range(252)]
    import numpy as np

    rng = np.random.default_rng(42)
    values = [100.0]
    for _ in range(251):
        values.append(values[-1] * (1 + rng.standard_normal() * 0.01))
    return pl.DataFrame({"date": dates, "value": values}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


@pytest.fixture
def price_series_pair(price_series):
    """Two correlated price series on the same dates."""
    import numpy as np

    rng = np.random.default_rng(7)
    dates = price_series["date"].to_list()
    values = [100.0]
    for _ in range(251):
        values.append(values[-1] * (1 + rng.standard_normal() * 0.01))
    s2 = pl.DataFrame({"date": dates, "value": values}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return price_series, s2


@pytest.fixture
def known_series():
    """Small hand-calculated series for exact numerical assertions."""
    dates = [dt.date(2024, 1, i + 1) for i in range(5)]
    values = [10.0, 20.0, 15.0, 25.0, 20.0]
    return pl.DataFrame({"date": dates, "value": values}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


@pytest.fixture
def empty_series():
    return pl.DataFrame({"date": [], "value": []}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
