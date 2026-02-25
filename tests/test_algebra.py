import datetime as dt
import math

import polars as pl
import pytest

from qtk.ts.algebra import (
    abs_,
    add,
    and_,
    ceil,
    divide,
    exp,
    filter_,
    floor,
    floordiv,
    geometrically_aggregate,
    if_,
    log,
    multiply,
    not_,
    or_,
    power,
    sqrt,
    subtract,
    weighted_sum,
    FilterOperator,
)


def _s(dates_values):
    dates, values = zip(*dates_values)
    return pl.DataFrame({"date": list(dates), "value": [float(v) for v in values]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


D1 = dt.date(2024, 1, 1)
D2 = dt.date(2024, 1, 2)
D3 = dt.date(2024, 1, 3)

X = _s([(D1, 4), (D2, 9), (D3, 16)])
Y = _s([(D1, 2), (D2, 3), (D3, 4)])


def test_add_series():
    result = add(X, Y)
    assert result["value"].to_list() == [6.0, 12.0, 20.0]


def test_add_scalar():
    result = add(X, 1)
    assert result["value"].to_list() == [5.0, 10.0, 17.0]


def test_subtract():
    result = subtract(X, Y)
    assert result["value"].to_list() == [2.0, 6.0, 12.0]


def test_multiply():
    result = multiply(X, Y)
    assert result["value"].to_list() == [8.0, 27.0, 64.0]


def test_divide():
    result = divide(X, Y)
    assert result["value"].to_list() == [2.0, 3.0, 4.0]


def test_floordiv():
    a = _s([(D1, 10.0), (D2, 7.0)])
    b = _s([(D1, 3.0), (D2, 2.0)])
    result = floordiv(a, b)
    assert result["value"].to_list() == [3.0, 3.0]


def test_exp():
    result = exp(_s([(D1, 0.0), (D2, 1.0)]))
    assert abs(result["value"][0] - 1.0) < 1e-10
    assert abs(result["value"][1] - math.e) < 1e-10


def test_log():
    result = log(_s([(D1, 1.0), (D2, math.e)]))
    assert abs(result["value"][0] - 0.0) < 1e-10
    assert abs(result["value"][1] - 1.0) < 1e-10


def test_power():
    result = power(X, 2)
    assert result["value"].to_list() == [16.0, 81.0, 256.0]


def test_sqrt():
    result = sqrt(X)
    assert result["value"].to_list() == [2.0, 3.0, 4.0]


def test_abs_():
    s = _s([(D1, -3.0), (D2, 5.0)])
    assert abs_(s)["value"].to_list() == [3.0, 5.0]


def test_floor():
    s = _s([(D1, 3.7), (D2, 2.1)])
    result = floor(s, 3.0)
    assert result["value"].to_list() == [3.7, 3.0]


def test_ceil():
    s = _s([(D1, 3.7), (D2, 2.1)])
    result = ceil(s, 3.0)
    assert result["value"].to_list() == [3.0, 2.1]


def test_filter_gt():
    # GREATER removes values greater than threshold; remaining values are <= 8
    result = filter_(X, FilterOperator.GREATER, 8)
    assert all(v <= 8 for v in result["value"].to_list())


def test_filter_lt():
    # LESS removes values less than threshold; remaining values are >= 10
    result = filter_(X, FilterOperator.LESS, 10)
    assert all(v >= 10 for v in result["value"].to_list())


def test_filter_dates():
    # N_EQUALS keeps only dates in the list
    result = filter_dates(X, FilterOperator.N_EQUALS, dates=[D1, D3])
    assert result.height == 2


def test_and_():
    flags = _s([(D1, 1.0), (D2, 0.0), (D3, 1.0)])
    flags2 = _s([(D1, 1.0), (D2, 1.0), (D3, 0.0)])
    result = and_(flags, flags2)
    assert result["value"].to_list() == [1.0, 0.0, 0.0]


def test_or_():
    flags = _s([(D1, 1.0), (D2, 0.0), (D3, 0.0)])
    flags2 = _s([(D1, 0.0), (D2, 0.0), (D3, 1.0)])
    result = or_(flags, flags2)
    assert result["value"].to_list() == [1.0, 0.0, 1.0]


def test_not_():
    flags = _s([(D1, 1.0), (D2, 0.0)])
    result = not_(flags)
    assert result["value"].to_list() == [0.0, 1.0]


def test_weighted_sum():
    result = weighted_sum([X, Y], [1.0, 2.0])
    # At D1: 4*1 + 2*2 = 8
    assert result.filter(pl.col("date") == D1)["value"][0] == 8.0


def test_geometrically_aggregate():
    rets = _s([(D1, 0.1), (D2, 0.2), (D3, -0.05)])
    result = geometrically_aggregate(rets)
    # (1.1)*(1.2)*(0.95) - 1 = 1.254 - 1 = 0.254
    expected = 1.1 * 1.2 * 0.95 - 1
    assert abs(result["value"][-1] - expected) < 1e-10


from qtk.ts.algebra import filter_dates
