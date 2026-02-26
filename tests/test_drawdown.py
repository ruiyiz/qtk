"""Tests for ts/drawdown.py."""

import datetime as dt
import math

import polars as pl
import pytest

from qtk.ts.drawdown import (
    average_drawdown,
    average_drawdown_length,
    average_recovery,
    conditional_drawdown,
    drawdown_deviation,
    drawdowns,
    find_drawdowns,
    sort_drawdowns,
)


def _s(vals, start=dt.date(2024, 1, 1)):
    dates = [start + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"date": dates, "value": [float(v) for v in vals]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def test_drawdowns_flat():
    """Constant series has zero drawdown everywhere."""
    s = _s([100, 100, 100, 100])
    dd = drawdowns(s)
    assert all(v == pytest.approx(0.0) for v in dd["value"].to_list())


def test_drawdowns_monotone_up():
    """Monotonically rising series has zero drawdown."""
    s = _s([100, 110, 120, 130])
    dd = drawdowns(s)
    assert all(v == pytest.approx(0.0) for v in dd["value"].to_list())


def test_drawdowns_known():
    """
    Peak = 120, then drop to 90: dd = 90/120 - 1 = -0.25.
    Hand-calculated.
    """
    s = _s([100, 120, 90])
    dd = drawdowns(s)
    assert dd["value"][0] == pytest.approx(0.0)
    assert dd["value"][1] == pytest.approx(0.0)
    assert dd["value"][2] == pytest.approx(90.0 / 120.0 - 1.0)


def test_find_drawdowns_one_event():
    """
    Series: 100 -> 110 -> 90 -> 110  (one complete drawdown event)
    Peak at 110, trough at 90, recovers to 110.
    """
    s = _s([100, 110, 90, 110])
    events = find_drawdowns(s)
    assert len(events) == 1
    ev = events[0]
    assert ev["depth"] == pytest.approx(90.0 / 110.0 - 1.0)
    assert ev["recovery"] is not None


def test_find_drawdowns_ongoing():
    """Series that never recovers has end=None."""
    s = _s([100, 80, 60])
    events = find_drawdowns(s)
    assert len(events) == 1
    assert events[0]["end"] is None
    assert events[0]["recovery"] is None


def test_sort_drawdowns_ordering():
    """Worst drawdown should come first."""
    s = _s([100, 90, 100, 80, 100])
    sorted_events = sort_drawdowns(s)
    depths = [e["depth"] for e in sorted_events]
    assert depths == sorted(depths)


def test_average_drawdown():
    """
    Two complete drawdowns: -10% and -20%.
    avg drawdown = (-0.1 + -0.2) / 2 = -0.15
    """
    s = _s([100, 90, 100, 80, 100])
    avg = average_drawdown(s)
    assert avg == pytest.approx(-0.15, rel=1e-3)


def test_average_drawdown_no_drawdown():
    s = _s([100, 110, 120])
    assert average_drawdown(s) == 0.0


def test_average_drawdown_length():
    s = _s([100, 90, 100])
    length = average_drawdown_length(s)
    assert length > 0


def test_average_recovery():
    s = _s([100, 90, 100])
    r = average_recovery(s)
    assert r > 0


def test_drawdown_deviation_no_events():
    s = _s([100, 110, 120])
    assert drawdown_deviation(s) == 0.0


def test_conditional_drawdown():
    """CDaR at p=1.0 should be <= max drawdown (covers all negative dd values)."""
    s = _s([100, 90, 95, 80, 85])
    cdar = conditional_drawdown(s, p=1.0)
    assert cdar <= 0.0
    cdar_tail = conditional_drawdown(s, p=0.1)
    assert cdar_tail <= cdar


def test_conditional_drawdown_invalid_p():
    s = _s([100, 90])
    with pytest.raises(Exception):
        conditional_drawdown(s, p=0.0)
