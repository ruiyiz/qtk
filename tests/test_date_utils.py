import datetime as dt

import pytest

from qtk.date_utils import DayCountConvention, PaymentFrequency, day_count_fraction, has_feb_29


def test_has_feb_29_leap():
    assert has_feb_29(dt.date(2020, 1, 1), dt.date(2020, 3, 1))


def test_has_feb_29_no_leap():
    assert not has_feb_29(dt.date(2021, 1, 1), dt.date(2021, 3, 1))


def test_day_count_actual_360():
    start = dt.date(2023, 1, 1)
    end = dt.date(2023, 4, 1)
    frac = day_count_fraction(start, end, DayCountConvention.ACTUAL_360)
    assert abs(frac - 90 / 360) < 1e-10


def test_day_count_actual_365f():
    start = dt.date(2023, 1, 1)
    end = dt.date(2023, 7, 1)
    frac = day_count_fraction(start, end, DayCountConvention.ACTUAL_365F)
    assert abs(frac - 181 / 365) < 1e-10


def test_day_count_one_one():
    frac = day_count_fraction(
        dt.date(2020, 1, 1), dt.date(2021, 1, 1), DayCountConvention.ONE_ONE
    )
    assert frac == 1


def test_day_count_actual_365l_non_annual():
    start = dt.date(2020, 1, 1)
    end = dt.date(2020, 12, 31)
    frac = day_count_fraction(
        start, end, DayCountConvention.ACTUAL_365L, PaymentFrequency.MONTHLY
    )
    # end.year 2020 is leap year
    assert abs(frac - 365 / 366) < 1e-10
