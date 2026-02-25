import datetime as dt

import polars as pl
import pytest

from qtk.ts.helper import (
    Window,
    _to_polars_duration,
    _to_timedelta,
    apply_ramp,
    normalize_window,
)
from qtk.errors import QtkValueError


def _make_df(n=10):
    dates = [dt.date(2024, 1, i + 1) for i in range(n)]
    values = list(range(n))
    return pl.DataFrame({"date": dates, "value": [float(v) for v in values]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def test_to_polars_duration_month():
    assert _to_polars_duration("1m") == "1mo"


def test_to_polars_duration_year():
    assert _to_polars_duration("2y") == "2y"


def test_to_polars_duration_day():
    assert _to_polars_duration("5d") == "5d"


def test_to_polars_duration_invalid():
    with pytest.raises(QtkValueError):
        _to_polars_duration("bad")


def test_to_timedelta_days():
    td = _to_timedelta("7d")
    assert td == dt.timedelta(days=7)


def test_normalize_window_int():
    df = _make_df(20)
    w = normalize_window(df, 5)
    assert w.w == 5 and w.r == 5


def test_normalize_window_none():
    df = _make_df(10)
    w = normalize_window(df, None)
    assert w.w == 10


def test_apply_ramp_int():
    df = _make_df(10)
    w = Window(10, 3)
    result = apply_ramp(df, w)
    assert result.height == 7


def test_apply_ramp_zero_ramp():
    df = _make_df(10)
    w = Window(10, 0)
    result = apply_ramp(df, w)
    assert result.height == 10


def test_apply_ramp_empty():
    df = pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Float64})
    w = Window(5, 5)
    result = apply_ramp(df, w)
    assert result.is_empty()
