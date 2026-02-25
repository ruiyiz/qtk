import datetime as dt

import polars as pl
import pytest

from qtk.ts.analysis import (
    LagMode,
    ThresholdType,
    compare,
    count,
    diff,
    first,
    lag,
    last,
    last_value,
    repeat,
    smooth_spikes,
)
from qtk.ts.helper import Interpolate


def _s(vals, start=dt.date(2024, 1, 1)):
    dates = [start + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"date": dates, "value": [float(v) for v in vals]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def test_smooth_spikes_removes_spike():
    s = _s([10, 10, 100, 10, 10])
    result = smooth_spikes(s, 0.5)
    # Spike at index 2 (value 100) should be replaced by avg(10, 10) = 10
    assert result["value"][1] == pytest.approx(10.0)


def test_smooth_spikes_short_series():
    result = smooth_spikes(_s([10, 20]), 0.5)
    assert result.is_empty()


def test_repeat_forward_fill():
    sparse = pl.DataFrame({
        "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 3)],
        "value": [1.0, 3.0],
    }).cast({"date": pl.Date, "value": pl.Float64})
    result = repeat(sparse, 1)
    assert result.height == 3
    assert result.filter(pl.col("date") == dt.date(2024, 1, 2))["value"][0] == 1.0


def test_repeat_downsample():
    s = _s([1, 2, 3, 4, 5, 6])
    result = repeat(s, 2)
    # Every 2nd row starting from first
    assert result["date"][1] == dt.date(2024, 1, 3)


def test_first():
    s = _s([10, 20, 30])
    result = first(s)
    assert all(v == 10.0 for v in result["value"].to_list())


def test_last():
    s = _s([10, 20, 30])
    result = last(s)
    assert all(v == 30.0 for v in result["value"].to_list())


def test_last_value():
    s = _s([10, 20, 30])
    assert last_value(s) == 30.0


def test_last_value_empty():
    with pytest.raises(Exception):
        last_value(pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Float64}))


def test_count():
    s = _s([1, 2, 3, 4, 5])
    result = count(s)
    assert result["value"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_diff_obs_1():
    s = _s([10, 20, 30])
    result = diff(s, 1)
    assert result["value"].to_list() == [10.0, 10.0]


def test_diff_obs_2():
    s = _s([10, 20, 30, 40])
    result = diff(s, 2)
    assert result["value"].to_list() == [20.0, 20.0]


def test_compare_gt():
    a = _s([10, 5, 10])
    b = _s([5, 10, 10])
    result = compare(a, b)
    assert result["value"].to_list() == [1.0, -1.0, 0.0]


def test_lag_int_truncate():
    s = _s([1, 2, 3, 4])
    result = lag(s, 2, LagMode.TRUNCATE)
    assert result["value"].to_list() == [1.0, 2.0]
    assert result["date"].to_list() == [dt.date(2024, 1, 3), dt.date(2024, 1, 4)]


def test_lag_int_extend():
    s = _s([1, 2, 3])
    result = lag(s, 1, LagMode.EXTEND)
    # Extends by 1 day; values shift forward
    assert result["value"].to_list() == [1.0, 2.0, 3.0]
    assert result["date"][-1] == dt.date(2024, 1, 4)


def test_lag_tenor_truncate():
    start = dt.date(2022, 1, 1)
    dates = [start + dt.timedelta(days=i * 30) for i in range(12)]
    values = list(range(1, 13))
    s = pl.DataFrame({"date": dates, "value": [float(v) for v in values]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    result = lag(s, "1m", LagMode.TRUNCATE)
    # Shifted dates should all be <= last original date
    assert all(d <= dates[-1] for d in result["date"].to_list())


def test_lag_empty():
    empty = pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Float64})
    assert lag(empty, 1).is_empty()
