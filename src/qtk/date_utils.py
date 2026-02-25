# Copyright 2019 Goldman Sachs.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# -----------------------------------------------------------------------
# MODIFICATION NOTICE (Apache License 2.0, Section 4b)
# This file has been modified from the original gs-quant source.
# Original source: https://github.com/goldmansachs/gs-quant
# Original copyright: Copyright 2019 Goldman Sachs.
# Modifications:
#   - Removed Goldman Sachs Marquee API dependencies (GsCalendar, PricingLocation)
#   - Removed functions that depend on GsCalendar: is_business_day,
#     business_day_offset, business_day_count, date_range, prev_business_date, today
#   - Ported from pandas.Series to polars.Series / pl.DataFrame
#   - Patched NumPy <2.0 deprecated APIs for NumPy >=2.0 compatibility
#   All mathematical logic is unchanged from the original.
# -----------------------------------------------------------------------

import calendar as cal
import datetime as dt
from enum import Enum, IntEnum


class PaymentFrequency(IntEnum):
    """Payment frequency enumeration

    Provides an enumeration of different payment frequencies used to to discount cashflows and accrue interest

    """

    DAILY = 252
    WEEKLY = 52
    SEMI_MONTHLY = 26
    MONTHLY = 12
    SEMI_QUARTERLY = 6
    QUARTERLY = 4
    TRI_ANNUALLY = 3
    SEMI_ANNUALLY = 2
    ANNUALLY = 1


class DayCountConvention(Enum):
    """Day Count Convention enumeration

    Provides an enumeration of different day count conventions for determining how interest accrues over payment periods
    for financial securities

    """

    # Actual/360: Number of days between dates divided by 360
    ACTUAL_360 = "ACTUAL_360"

    # Actual/364: Number of days between dates divided by 364
    ACTUAL_364 = "ACTUAL_364"

    # Actual/365_25: Number of days between dates divided by 365.25
    ACTUAL_365_25 = "ACTUAL_365_25"

    # Actual/365 FIXED: Number of days between dates divided by 365
    ACTUAL_365F = "ACTUAL_365F"

    # Actual/365 LEAP: Number of days between dates divided by 365 or 366 in leap years
    ACTUAL_365L = "ACTUAL_365L"

    # ONE_ONE: Always returns a day count fraction of 1
    ONE_ONE = "ONE_ONE"


def has_feb_29(start: dt.date, end: dt.date) -> bool:
    """
    Determine if date range has a leap day (29Feb)

    :param start: first date
    :param end: second date

    **Usage**

    Determine if a given date range contains a leap day (Feb 29). Used for various day count convention calculations
    which alter behaviour for leap years. Start date is exclusive and end date is inclusive

    **Examples**

    Determine if a given date range contains 29Feb

    >>> start = dt.date(2020, 1, 1)
    >>> end = dt.date(2020, 3, 15)
    >>> has_feb_29(start, end)
    """
    for x in range(1, (end - start).days + 1):
        date = start + dt.timedelta(days=x)
        if date.month == 2 and date.day == 29:
            return True
    return False


def day_count_fraction(
    start: dt.date,
    end: dt.date,
    convention: DayCountConvention = DayCountConvention.ACTUAL_360,
    frequency: PaymentFrequency = PaymentFrequency.MONTHLY,
) -> float:
    """
    Compute day count fraction between dates

    :param start: first date
    :param end: second date
    :param convention: day count convention
    :param frequency: payment frequency of instrument
    :return: day count fraction between dates per convention

    **Usage**

    Compute day count fraction between dates, based on the value of *convention*. For more information on the available
    day count conventions, see the
    `Day Count Conventions <https://developer.gs.com/docs/gsquant/guides/Dates/1-day-count-conventions>`_ guide.

    **Examples**

    Compute day count fraction between two dates using Actual/360 convention:

    >>> start = dt.date(2015, 11, 12)
    >>> end = dt.date(2017, 12, 15)
    >>> day_count_fraction(start, end, DayCountConvention.ACTUAL_360)

    """
    if convention == DayCountConvention.ACTUAL_360:
        return (end - start).days / 360
    elif convention == DayCountConvention.ACTUAL_364:
        return (end - start).days / 364
    elif convention == DayCountConvention.ACTUAL_365F:
        return (end - start).days / 365
    elif convention == DayCountConvention.ACTUAL_365L:
        if frequency == PaymentFrequency.ANNUALLY:
            days_in_year = 366 if has_feb_29(start, end) else 365
        else:
            days_in_year = 366 if cal.isleap(end.year) else 365
        return (end - start).days / days_in_year
    elif convention == DayCountConvention.ACTUAL_365_25:
        return (end - start).days / 365.25
    elif convention == DayCountConvention.ONE_ONE:
        return 1
    else:
        raise ValueError("Unknown day count convention: " + convention.value)
