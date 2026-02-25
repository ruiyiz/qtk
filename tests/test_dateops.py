import datetime as dt

import polars as pl
import pytest

from qtk.ts.dateops import (
    AggregateFunction,
    AggregatePeriod,
    align,
    append,
    bucketize,
    date_range,
    day,
    interpolate,
    month,
    prepend,
    union,
    value,
    weekday,
    year,
)
from qtk.ts.helper import Interpolate


def _make(dates_values):
    dates, values = zip(*dates_values)
    return pl.DataFrame({"date": list(dates), "value": [float(v) for v in values]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


X = _make([
    (dt.date(2024, 1, 1), 10),
    (dt.date(2024, 1, 3), 30),
    (dt.date(2024, 1, 5), 50),
])

Y = _make([
    (dt.date(2024, 1, 2), 20),
    (dt.date(2024, 1, 3), 35),
    (dt.date(2024, 1, 4), 40),
])


def test_align_intersect():
    a, b = align(X, Y, Interpolate.INTERSECT)
    assert a["date"].to_list() == [dt.date(2024, 1, 3)]
    assert a["value"][0] == 30.0
    assert b["value"][0] == 35.0


def test_align_nan():
    a, b = align(X, Y, Interpolate.NAN)
    assert a.height == b.height
    assert a.height == 5  # union of {1,3,5} and {2,3,4}


def test_align_zero():
    a, b = align(X, Y, Interpolate.ZERO)
    # dates without a value get 0 in that series
    a_date1 = a.filter(pl.col("date") == dt.date(2024, 1, 2))["value"][0]
    assert a_date1 == 0.0


def test_align_step():
    a, b = align(X, Y, Interpolate.STEP)
    # date 2024-01-2: X value should be forward-filled from 2024-01-1 -> 10
    a_d2 = a.filter(pl.col("date") == dt.date(2024, 1, 2))["value"][0]
    assert a_d2 == 10.0


def test_value_exact():
    v = value(X, dt.date(2024, 1, 1))
    assert v == 10.0


def test_value_step_interpolation():
    v = value(X, dt.date(2024, 1, 2), Interpolate.STEP)
    assert v == 10.0  # forward fill from Jan 1


def test_day_extractor():
    result = day(X)
    assert result["value"].to_list() == [1.0, 3.0, 5.0]


def test_month_extractor():
    result = month(X)
    assert all(v == 1.0 for v in result["value"].to_list())


def test_year_extractor():
    result = year(X)
    assert all(v == 2024.0 for v in result["value"].to_list())


def test_weekday_extractor():
    result = weekday(X)
    # 2024-01-01 is Monday (0-indexed in our port)
    assert result["value"][0] == 0.0


def test_date_range_filter():
    result = date_range(X, dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert result.height == 2
    assert result["date"][0] == dt.date(2024, 1, 1)


def test_append_series():
    extra = _make([(dt.date(2024, 1, 6), 60), (dt.date(2024, 1, 7), 70)])
    result = append([X, extra])
    assert result.height == 5


def test_prepend_series():
    extra = _make([(dt.date(2023, 12, 30), 5)])
    result = prepend([extra, X])
    assert result.height == 4
    assert result["date"][0] == dt.date(2023, 12, 30)


def test_union_series():
    result = union([X, Y])
    # union of {Jan1,3,5} and {Jan2,3,4} = {Jan1,2,3,4,5}
    assert result.height == 5


def test_bucketize():
    # monthly bucket
    monthly = _make([
        (dt.date(2024, 1, 10), 10),
        (dt.date(2024, 1, 20), 20),
        (dt.date(2024, 2, 5), 30),
    ])
    result = bucketize(monthly, AggregateFunction.MEAN, AggregatePeriod.MONTH)
    assert result.height == 2


def test_interpolate_step():
    sparse = _make([(dt.date(2024, 1, 1), 1), (dt.date(2024, 1, 5), 5)])
    dense_dates = [dt.date(2024, 1, i + 1) for i in range(5)]
    result = interpolate(sparse, dense_dates, Interpolate.STEP)
    assert result.height == 5
    # Jan2 should be 1 (forward fill from Jan1)
    assert result.filter(pl.col("date") == dt.date(2024, 1, 2))["value"][0] == 1.0
