# Copyright 2018 Goldman Sachs.
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
# Original copyright: Copyright 2018 Goldman Sachs.
# Modifications:
#   - Removed Goldman Sachs Marquee API dependencies
#   - Ported from pandas.Series to polars.Series / pl.DataFrame
#   - Patched NumPy <2.0 deprecated APIs for NumPy >=2.0 compatibility
#   - Removed align_calendar (depends on GsCalendar Marquee API)
#   - Named dateops.py to avoid shadowing stdlib datetime module
#   All mathematical logic is unchanged from the original.
# -----------------------------------------------------------------------

"""
Date and time manipulation for timeseries, including date or time shifting, calendar operations, curve alignment and
interpolation operations. Includes sampling operations based on date or time manipulation
"""

import datetime as dt
from enum import Enum
from numbers import Real
from typing import Any, List, Union

import numpy as np
import polars as pl

from qtk.date_utils import DayCountConvention, PaymentFrequency, day_count_fraction
from qtk.errors import QtkTypeError, QtkValueError
from qtk.ts.helper import Interpolate, Window, _create_enum

__all__ = [
    "AggregateFunction",
    "AggregatePeriod",
    "align",
    "interpolate",
    "value",
    "day",
    "month",
    "year",
    "quarter",
    "weekday",
    "day_count_fractions",
    "date_range",
    "append",
    "prepend",
    "union",
    "bucketize",
    "day_count",
    "day_countdown",
]

AggregateFunction: Union[type, Enum, Any] = _create_enum(
    "AggregateFunction", ["max", "min", "mean", "sum", "first", "last"]
)
AggregatePeriod = _create_enum("AggregatePeriod", ["week", "month", "quarter", "year"])


def align(
    x: Union[pl.DataFrame, Real],
    y: Union[pl.DataFrame, Real],
    method: Interpolate = Interpolate.INTERSECT,
) -> List[Union[pl.DataFrame, Real]]:
    """
    Align dates of two series or scalars

    :param x: first timeseries or scalar
    :param y: second timeseries or scalar
    :param method: interpolation method (default: intersect). Only used when both x and y are timeseries
    :return: timeseries with specified dates or two scalars from the input

    **Usage**

    Align the dates of two series using the specified interpolation method. Returns two series with dates based on the
    method of interpolation, for example, can be used to intersect the dates of two series, union dates with a defined
    manner to compute missing values.

    Interpolation methods:

    =========   ========================================================================
    Type        Behavior
    =========   ========================================================================
    intersect   Resultant series only have values on the intersection of dates /times.
    nan         Resultant series have values on the union of dates /times. Values will
                be NaN for dates or times only present in the other series
    zero        Resultant series have values on the union of  dates / times. Values will
                be zero for dates or times only present in the other series
    step        Resultant series have values on the union of  dates / times. Each series
                will use the value of the previous valid point if requested date does
                not exist. Values prior to the first date will be equivalent to the
                first available value
    time        Resultant series have values on the union of dates / times. Missing
                values surrounded by valid values will be interpolated given length of
                interval. Input series must use DateTimeIndex.
    =========   ========================================================================

    **Examples**

    Stepwize interpolation of series based on dates in second series:

    >>> a = generate_series(100)
    >>> b = generate_series(100)
    >>> align(a, b)

    **See also**

    :func:`sub`
    """
    if isinstance(x, Real) and isinstance(y, Real):
        return [x, y]
    if isinstance(x, Real):
        return [
            pl.DataFrame({"date": y["date"].to_list(), "value": [float(x)] * y.height}).cast(
                {"date": pl.Date, "value": pl.Float64}
            ),
            y,
        ]
    if isinstance(y, Real):
        return [
            x,
            pl.DataFrame({"date": x["date"].to_list(), "value": [float(y)] * x.height}).cast(
                {"date": pl.Date, "value": pl.Float64}
            ),
        ]

    if method == Interpolate.INTERSECT:
        joined = x.join(y, on="date", how="inner", suffix="_y")
        return [
            joined.select("date", "value"),
            joined.select("date", pl.col("value_y").alias("value")),
        ]

    if method == Interpolate.NAN:
        joined = x.join(y, on="date", how="full", suffix="_y", coalesce=True).sort("date")
        return [
            joined.select("date", "value"),
            joined.select("date", pl.col("value_y").alias("value")),
        ]

    if method == Interpolate.ZERO:
        joined = x.join(y, on="date", how="full", suffix="_y", coalesce=True).sort("date")
        return [
            joined.select("date", pl.col("value").fill_null(0.0)),
            joined.select("date", pl.col("value_y").fill_null(0.0).alias("value")),
        ]

    if method == Interpolate.STEP:
        joined = x.join(y, on="date", how="full", suffix="_y", coalesce=True).sort("date")
        return [
            joined.select("date", pl.col("value").forward_fill().backward_fill()),
            joined.select("date", pl.col("value_y").forward_fill().backward_fill().alias("value")),
        ]

    if method == Interpolate.TIME:
        joined = x.join(y, on="date", how="full", suffix="_y", coalesce=True).sort("date")
        return [
            joined.select(
                "date",
                pl.col("value").interpolate_by(pl.col("date").cast(pl.Int32)),
            ),
            joined.select(
                "date",
                pl.col("value_y").interpolate_by(pl.col("date").cast(pl.Int32)).alias("value"),
            ),
        ]

    raise QtkValueError("Unknown intersection type: " + str(method))


def interpolate(
    x: pl.DataFrame,
    dates: Union[List[dt.date], pl.DataFrame, None] = None,
    method: Interpolate = Interpolate.INTERSECT,
) -> pl.DataFrame:
    """
    Interpolate over specified dates or times

    :param x: timeseries to interpolate
    :param dates: array of dates or another series to interpolate
    :param method: interpolation method (default: intersect)
    :return: timeseries with specified dates

    **Usage**

    Interpolate the series X over the dates specified by the dates parameter. This can be an array of dates or another
    series, in which case the index of the series will be used to specify dates

    Interpolation methods:

    =========   ========================================================================
    Type        Behavior
    =========   ========================================================================
    intersect   Resultant series only has values on the intersection of dates /times.
                Will only contain intersection of valid dates / times in the series
    nan         Resultant series only has values on the intersection of dates /times.
                Value will be NaN for dates not present in the series
    zero        Resultant series has values on all requested dates / times. The series
                will have a value of zero where the requested date or time was not
                present in the series
    step        Resultant series has values on all requested dates / times. The series
                will use the value of the previous valid point if requested date does
                not exist. Values prior to the first date will be equivalent to the
                first available value
    =========   ========================================================================

    **Examples**

    Stepwize interpolation of series based on dates in second series:

    >>> a = generate_series(100)
    >>> b = generate_series(100)
    >>> interpolate(a, b, Interpolate.INTERSECT)

    **See also**

    :func:`sub`
    """
    if dates is None:
        return x

    if isinstance(dates, pl.DataFrame):
        target_dates = dates["date"]
    else:
        target_dates = pl.Series("date", dates, dtype=pl.Date)

    target = pl.DataFrame({"date": target_dates})

    if method == Interpolate.INTERSECT:
        return x.join(target, on="date", how="inner")

    if method == Interpolate.NAN:
        return target.join(x, on="date", how="left").sort("date")

    if method == Interpolate.ZERO:
        return target.join(x, on="date", how="left").with_columns(
            pl.col("value").fill_null(0.0)
        ).sort("date")

    if method == Interpolate.STEP:
        # Include x's own dates so forward_fill has prior values to propagate
        all_dates = (
            pl.concat([x.select("date"), target.select("date")])
            .unique(subset=["date"])
            .sort("date")
        )
        filled = all_dates.join(x, on="date", how="left").sort("date")
        filled = filled.with_columns(pl.col("value").forward_fill().backward_fill())
        return filled.join(target, on="date", how="inner")

    raise QtkValueError("Unknown intersection type: " + str(method))


def value(
    x: pl.DataFrame,
    date: dt.date,
    method: Interpolate = Interpolate.STEP,
) -> Union[float, None]:
    """
    Value at specified date or time

    :param x: timeseries
    :param date: requested date or time
    :param method: interpolation method (default: step)
    :return: value at specified date or time

    **Usage**

    Returns the value of series X at the specified date:

    :math:`Y_t = X_{date}`

    If the requested date or time is not present in the series, the value function will return the value from the
    previous available date or time by default. Caller can specify other interpolation styles via the method param:

    Interpolation methods:

    =========   ========================================================================
    Type        Behavior
    =========   ========================================================================
    intersect   Only returns a value for valid dates
    nan         Value will be NaN for dates not present in the series
    zero        Value will be zero for dates not present in the series
    step        Value of the previous valid point if requested date does not exist.
                Values prior to the first date will be equivalent to the first available
                value
    =========   ========================================================================

    **Examples**

    Value of series on 5Mar18:

    >>> a = generate_series(100)
    >>> value(a, dt.date(2019, 1, 3))

    **See also**

    :func:`interpolate`
    """
    result = interpolate(x, [date], method)
    return None if result.is_empty() else result["value"][0]


def day(x: pl.DataFrame) -> pl.DataFrame:
    """
    Day of each value in series

    :param x: time series
    :return: day of observations

    **Usage**

    Returns the day as a numeric value for each observation in the series:

    :math:`Y_t = day(t)`

    Day of the time or date is the integer day number within the month, e.g. 1-31

    **Examples**

    Day for observations in series:

    >>> series = generate_series(100)
    >>> days = day(series)

    **See also**

    :func:`month` :func:`year`

    """
    return x.select("date", pl.col("date").dt.day().cast(pl.Int64).alias("value"))


def month(x: pl.DataFrame) -> pl.DataFrame:
    """
    Month of each value in series

    :param x: time series
    :return: month of observations

    **Usage**

    Returns the month as a numeric value for each observation in the series:

    :math:`Y_t = month(t)`

    Month of the time or date is the integer month number, e.g. 1-12

    **Examples**

    Day for observations in series:

    >>> series = generate_series(100)
    >>> days = month(series)

    **See also**

    :func:`day` :func:`year`

    """
    return x.select("date", pl.col("date").dt.month().cast(pl.Int64).alias("value"))


def year(x: pl.DataFrame) -> pl.DataFrame:
    """
    Year of each value in series

    :param x: time series
    :return: year of observations

    **Usage**

    Returns the year as a numeric value for each observation in the series:

    :math:`Y_t = year(t)`

    Year of the time or date is the integer year number, e.g. 2019, 2020

    **Examples**

    Year for observations in series:

    >>> series = generate_series(100)
    >>> days = year(series)

    **See also**

    :func:`day` :func:`month`

    """
    return x.select("date", pl.col("date").dt.year().cast(pl.Int64).alias("value"))


def quarter(x: pl.DataFrame) -> pl.DataFrame:
    """
    Quarter of each value in series

    :param x: time series
    :return: quarter of observations

    **Usage**

    Returns the quarter as a numeric value for each observation in the series:

    :math:`Y_t = quarter(t)`

    Quarter of the time or date is the integer quarter number, e.g. 1, 2, 3, 4

    **Examples**

    Quarter for observations in series:

    >>> series = generate_series(100)
    >>> days = quarter(series)

    **See also**

    :func:`day` :func:`month`

    """
    return x.select("date", pl.col("date").dt.quarter().cast(pl.Int64).alias("value"))


def weekday(x: pl.DataFrame) -> pl.DataFrame:
    """
    Weekday of each value in series

    :param x: time series
    :return: weekday of observations

    **Usage**

    Returns the weekday as a numeric value for each observation in the series:

    :math:`Y_t = weekday(t)`

    Weekday of the time or date is the integer day of the week, e.g. 0-6, where 0 represents Monday

    **Examples**

    Weekday for observations in series:

    >>> series = generate_series(100)
    >>> days = weekday(series)

    **See also**

    :func:`day` :func:`month`

    """
    # polars weekday: 1=Monday...7=Sunday; subtract 1 to match pandas (0=Monday)
    return x.select("date", (pl.col("date").dt.weekday() - 1).cast(pl.Int64).alias("value"))


def day_count_fractions(
    dates: Union[List[dt.date], pl.DataFrame],
    convention: DayCountConvention = DayCountConvention.ACTUAL_360,
    frequency: PaymentFrequency = PaymentFrequency.MONTHLY,
) -> pl.DataFrame:
    """
    Day count fractions between dates in series

    :param dates: time series or array of dates
    :param convention: day count convention (default: Actual/360 ISDA)
    :param frequency: payment frequency of instrument (optional)
    :return: series of day count fractions

    **Usage**

    Returns the day count fraction between dates in the series

    :math:`Y_t = DCF(t_{-1}, t)`

    Default is Actual/360 per ISDA specification:

    :math:`Y_t = \\frac{Days(t_{-1}, t)}{360}`

    For a full list of available conventions, see
    `Day Count Conventions <https://developer.gs.com/docs/gsquant/guides/Dates/1-day-count-conventions>`_.
    For more information on day count conventions, see the
    `day count conventions <https://en.wikipedia.org/wiki/Day_count_convention>`_ page on Wikipedia

    **Examples**

    Weekday for observations in series:

    >>> series = generate_series(100)
    >>> days = day_count_fractions(series)

    **See also**

    :func:`day` :func:`month` :func:`year`

    """
    if isinstance(dates, pl.DataFrame):
        date_list = dates["date"].to_list()
    else:
        date_list = list(dates)

    if len(date_list) < 2:
        return pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Float64})

    dcfs = [None] + [
        day_count_fraction(date_list[i], date_list[i + 1], convention, frequency)
        for i in range(len(date_list) - 1)
    ]
    return pl.DataFrame({"date": date_list, "value": dcfs})


def date_range(
    x: pl.DataFrame,
    start_date: Union[dt.date, int],
    end_date: Union[dt.date, int],
    weekdays_only: bool = False,
) -> pl.DataFrame:
    """
    Create a time series from a (sub-)range of dates in an existing time series.

    :param x: time series
    :param start_date: start date for the sliced time series. If integer, number of observations after the first
    :param end_date: end date for the sliced time series. If integer, number of observations before the last
    :param weekdays_only: whether to include only weekdays in the sliced ranges
    :return: sliced time series

    **Usage**

    Returns a restricted ("sliced") time series based on start and end dates:

    :math:`Y_t = R_t |_{start < t < end}`

    **Examples**

    Slice the first and last week of a time series:

    >>> series = generate_series(100)
    >>> sliced_series = date_range(series,7,7)

    create time series with the absolute date:

    >>> date_range(series, dt.date(2021,1,1), dt.date(2021,12,30))

    **See also**

    :func:`day` :func: `lag`

    """
    if not isinstance(weekdays_only, bool):
        raise QtkTypeError('expected a boolean value for "weekdays_only"')

    if isinstance(start_date, int):
        start_date = x["date"][start_date]
    if isinstance(end_date, int):
        end_date = x["date"][-(1 + end_date)]

    result = x.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))

    if weekdays_only:
        result = result.filter(pl.col("date").dt.weekday() <= 5)  # Mon=1..Fri=5

    return result


def append(series: List[pl.DataFrame]) -> pl.DataFrame:
    """
    Append data series

    :param series: an array of timeseries
    :return: concatenated timeseries

    **Usage**

    For input series [:math:`x_1`, :math:`x_2`, ... , :math:`x_n`], takes data from series :math:`x_i` until
    that series runs out, then appends data from `x_{i+1}` until that series runs out.

    **Examples**

    Append two series:

    >>> x = generate_series(100)
    >>> y = generate_series(100)
    >>> append([x, y])

    **See also**

    :func:`prepend`

    """
    if not series:
        return pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Float64})
    res = series[0]
    for i in range(1, len(series)):
        cur = series[i]
        if res.is_empty():
            res = cur
            continue
        start = res["date"][-1]
        res = pl.concat([res, cur.filter(pl.col("date") > start)])
    return res


def prepend(x: List[pl.DataFrame]) -> pl.DataFrame:
    """
    Prepend data series

    :param x: an array of timeseries
    :return: concatenated timeseries

    **Usage**

    For input series [:math:`x_1`, :math:`x_2`, ... , :math:`X_n`], takes data from series :math:`X_i` until
    the first date for which :math:`X_{i+1}` has data, useful when a higher quality series has a shorter history
    than a lower quality series.

    **Examples**

    Prepend two series:

    >>> x = generate_series(100)
    >>> y = generate_series(100)
    >>> prepend([x, y])

    **See also**

    :func:`union`

    """
    if not x:
        return pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Float64})
    if len(x) == 1:
        return x[0]
    parts = []
    for i in range(len(x)):
        this = x[i]
        if i == len(x) - 1:
            parts.append(this)
        else:
            end = x[i + 1]["date"][0]
            parts.append(this.filter(pl.col("date") < end))
    return pl.concat(parts)


def union(x: List[pl.DataFrame]) -> pl.DataFrame:
    """
    Fill in missing dates or times of one series with another

    :param x: an array of timeseries
    :return: combined series

    **Usage**

    Starting from :math:`i=1`, takes points from series :math:`x_i`. Where points are missing from :math:`x_i`,
    returns points from :math:`x_{i+1}`.

    **Examples**

    Union of two series:

    >>> x = generate_series(100)
    >>> y = generate_series(100)
    >>> union([x, y])

    **See also**

    :func:`prepend`

    """
    if not x:
        return pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Float64})
    res = x[0]
    for series in x[1:]:
        joined = res.join(series, on="date", how="full", suffix="_fill", coalesce=True).sort("date")
        res = joined.with_columns(
            pl.when(pl.col("value").is_not_null())
            .then(pl.col("value"))
            .otherwise(pl.col("value_fill"))
            .alias("value")
        ).select("date", "value")
    return res


def bucketize(
    series: pl.DataFrame,
    aggregate_function: "AggregateFunction",
    period: "AggregatePeriod",
) -> pl.DataFrame:
    """
    Bucketize a series and apply aggregate function to each bucket

    :param series: input series
    :param aggregate_function: function to use for aggregating data in each bucket
    :param period: size of each bucket
    :return: output series

    **Usage**

    Bucketize a series and apply aggregate function to each bucket. The result will be indexed by the end date of each
    bucket.

    **Examples**

    Monthly mean of a series:

    >>> x = generate_series(100)
    >>> bucketize(x, AggregateFunction.MEAN, AggregatePeriod.MONTH)

    **See also**

    :func:`moving_average`
    """
    period_char = period.value[0].upper()
    period_map = {"W": "1w", "M": "1mo", "Q": "3mo", "Y": "1y"}
    every = period_map.get(period_char, "1mo")

    agg_name = aggregate_function.value
    agg_map = {
        "max": pl.col("value").max(),
        "min": pl.col("value").min(),
        "mean": pl.col("value").mean(),
        "sum": pl.col("value").sum(),
        "first": pl.col("value").first(),
        "last": pl.col("value").last(),
    }
    agg_expr = agg_map.get(agg_name, pl.col("value").mean())

    if series.is_empty():
        return series

    result = (
        series.sort("date")
        .group_by_dynamic("date", every=every, closed="right", label="right")
        .agg(agg_expr)
        .sort("date")
    )
    return result


def day_count(first: dt.date, second: dt.date) -> int:
    """
    Counts the number of business days between two dates. Monday through Friday are considered to be business days.

    :param first: first date
    :param second: second date
    :return: number of business days between first and second
    """
    if not (isinstance(first, dt.date) and isinstance(second, dt.date)):
        raise QtkValueError("inputs must be dates")
    return int(np.busday_count(first, second))


def day_countdown(
    end_date: dt.date,
    start_date: dt.date = None,
    business_days: bool = False,
) -> pl.DataFrame:
    """Create a series counting down the number of days from each date to ``end_date``.

    :param end_date: last date / countdown target
    :param start_date: first date in the generated series (default: today)
    :param business_days: whether to use business days (default False)
    :return: timeseries of day counts

    **Usage**

    The returned series is indexed by dates from ``start_date`` to ``end_date`` (inclusive). Values are integers giving
    the number of days between the index date and ``end_date``.

    If ``business_days`` is True, the index will include only Monday through Friday as business days and the
    counts will be business-day counts. If False (default), the index will include all
    calendar days and the counts will be calendar-day counts.

    **Examples**

    Calendar-day countdown from 7May21 to 17May21:
    >>> day_countdown(dt.date(2035, 5, 17), dt.date(2025, 5, 7))

    Business-day countdown (Mon-Fri only):
    >>> day_countdown(dt.date(2035, 5, 17), dt.date(2025, 5, 7), True)

    Calendar-day countdown starting today:
    >>> day_countdown(dt.date(2035, 5, 17))
    """
    if start_date is None:
        start_date = dt.date.today()

    if not isinstance(start_date, dt.date):
        raise QtkValueError("start_date must be a date")
    if not isinstance(end_date, dt.date):
        raise QtkValueError("end_date must be a date")

    if start_date > end_date:
        return pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Int64})

    if business_days:
        total_days = (end_date - start_date).days + 1
        all_dates = [start_date + dt.timedelta(days=i) for i in range(total_days)]
        biz_dates = [d for d in all_dates if d.weekday() < 5]
        if not biz_dates:
            return pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Int64})
        start_arr = np.array([d for d in biz_dates], dtype="datetime64[D]")
        end_arr = np.datetime64(end_date, "D")
        counts = np.busday_count(start_arr, end_arr).astype(np.int64)
        return pl.DataFrame({"date": biz_dates, "value": counts.tolist()})
    else:
        total_days = (end_date - start_date).days + 1
        all_dates = [start_date + dt.timedelta(days=i) for i in range(total_days)]
        start_arr = np.array(all_dates, dtype="datetime64[D]")
        end_arr = np.datetime64(end_date, "D")
        counts = (end_arr - start_arr).astype(np.int64)
        return pl.DataFrame({"date": all_dates, "value": counts.tolist()})
