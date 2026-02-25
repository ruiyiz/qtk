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
#   - Removed Marquee-specific decorators and utilities
#   All mathematical logic is unchanged from the original.
# -----------------------------------------------------------------------

"""
Timeseries analysis library contains functions used to analyze properties of timeseries,
including lagging, differencing, and other operations.
"""

import datetime as dt
import re
from enum import Enum
from numbers import Real
from typing import Union

import polars as pl

from qtk.errors import QtkValueError
from qtk.ts.dateops import align
from qtk.ts.helper import (
    Interpolate,
    Window,
    _to_polars_duration,
)

__all__ = [
    "ThresholdType",
    "LagMode",
    "smooth_spikes",
    "repeat",
    "first",
    "last",
    "last_value",
    "count",
    "diff",
    "compare",
    "lag",
]


class ThresholdType(str, Enum):
    percentage = "percentage"
    absolute = "absolute"


class LagMode(Enum):
    TRUNCATE = "truncate"
    EXTEND = "extend"


def smooth_spikes(
    x: pl.DataFrame,
    threshold: float,
    threshold_type: ThresholdType = ThresholdType.percentage,
) -> pl.DataFrame:
    """
    Smooth out the spikes of a series. If a point is larger/smaller than (1 +/- threshold)
    times both neighbors, replace it with the average of those neighbours.
    The first and last points are dropped.

    :param x: timeseries
    :param threshold: minimum increment to trigger filter
    :param threshold_type: type of threshold check
    :return: smoothed timeseries
    """
    if x.height < 3:
        return pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Float64})

    threshold_value = threshold if threshold_type == ThresholdType.absolute else (1 + threshold)

    def check_percentage(prev, curr, nxt, mult):
        curr_higher = curr > prev * mult and curr > nxt * mult
        curr_lower = prev > curr * mult and nxt > curr * mult
        return curr_higher or curr_lower

    def check_absolute(prev, curr, nxt, abs_val):
        curr_higher = curr > prev + abs_val and curr > nxt + abs_val
        curr_lower = prev > curr + abs_val and nxt > curr + abs_val
        return curr_higher or curr_lower

    check_fn = check_absolute if threshold_type == ThresholdType.absolute else check_percentage

    dates = x["date"].to_list()
    values = x["value"].to_list()
    result = list(values)

    for i in range(1, len(values) - 1):
        if check_fn(values[i - 1], values[i], values[i + 1], threshold_value):
            result[i] = (values[i - 1] + values[i + 1]) / 2

    return pl.DataFrame({"date": dates[1:-1], "value": result[1:-1]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def repeat(x: pl.DataFrame, n: int = 1) -> pl.DataFrame:
    """
    Repeat values for days where data is missing (forward fill), then optionally
    downsample so there are data points every n days.

    :param x: date-based timeseries
    :param n: desired frequency of output (days between observations)
    :return: forward-filled and optionally downsampled timeseries
    """
    if not 0 < n < 367:
        raise QtkValueError("n must be between 0 and 367")
    if x.is_empty():
        return x

    first_date = x["date"][0]
    last_date = x["date"][-1]

    all_dates = pl.DataFrame(
        {"date": pl.date_range(first_date, last_date, interval="1d", eager=True)}
    )
    filled = (
        all_dates.join(x, on="date", how="left")
        .sort("date")
        .with_columns(pl.col("value").forward_fill())
        .drop_nulls()
    )

    if n > 1:
        filled = filled[::n]

    return filled


def first(x: pl.DataFrame) -> pl.DataFrame:
    """
    First value of series broadcast across all dates.

    :param x: timeseries
    :return: timeseries where every value equals the first observation
    """
    if x.is_empty():
        return x
    first_val = x["value"][0]
    return x.with_columns(pl.lit(first_val).cast(pl.Float64).alias("value"))


def last(x: pl.DataFrame) -> pl.DataFrame:
    """
    Last value of series broadcast across all dates.

    :param x: timeseries
    :return: timeseries where every value equals the last non-null observation
    """
    if x.is_empty():
        return x
    non_null = x.drop_nulls("value")
    last_val = non_null["value"][-1] if not non_null.is_empty() else None
    return x.with_columns(pl.lit(last_val).cast(pl.Float64).alias("value"))


def last_value(x: pl.DataFrame) -> float:
    """
    Last value of series as a scalar.

    :param x: timeseries
    :return: last non-null value
    """
    if x.is_empty():
        raise QtkValueError("cannot get last value of an empty series")
    non_null = x.drop_nulls("value")
    if non_null.is_empty():
        raise QtkValueError("cannot get last value of an empty series")
    return non_null["value"][-1]


def count(x: pl.DataFrame) -> pl.DataFrame:
    """
    Cumulative count of non-null observations.

    :param x: timeseries
    :return: cumulative count series
    """
    return x.with_columns(
        pl.col("value").is_not_null().cast(pl.Int64).cum_sum().cast(pl.Float64).alias("value")
    )


def diff(x: pl.DataFrame, obs: Union[Window, int, str] = 1) -> pl.DataFrame:
    """
    Difference of series with given lag: R_t = X_t - X_{t-obs}.

    :param x: timeseries
    :param obs: number of observations to lag or tenor string e.g. '3d', '1w', '1m'
    :return: differenced timeseries
    """
    lagged = lag(x, obs, LagMode.TRUNCATE)
    xa, ya = align(x, lagged, Interpolate.INTERSECT)
    return xa.with_columns((pl.col("value") - ya["value"]).alias("value"))


def compare(
    x: Union[pl.DataFrame, Real],
    y: Union[pl.DataFrame, Real],
    method: Interpolate = Interpolate.STEP,
) -> Union[pl.DataFrame, Real]:
    """
    Compare two series: returns 1 where x > y, -1 where x < y, 0 where x == y.

    :param x: timeseries or scalar
    :param y: timeseries or scalar
    :param method: interpolation method when aligning two series
    :return: comparison signal series
    """
    if isinstance(x, Real) and isinstance(y, Real):
        return 1.0 if x > y else (-1.0 if x < y else 0.0)

    xa, ya = align(x, y, method)
    return xa.with_columns(
        (
            (pl.col("value") > ya["value"]).cast(pl.Float64)
            - (pl.col("value") < ya["value"]).cast(pl.Float64)
        ).alias("value")
    )


def lag(
    x: pl.DataFrame,
    obs: Union[Window, int, str] = 1,
    mode: LagMode = LagMode.EXTEND,
) -> pl.DataFrame:
    """
    Lag timeseries by a number of observations or a relative date.

    :param x: timeseries
    :param obs: non-zero integer or tenor string e.g. '1d', '1m', '1y'
    :param mode: TRUNCATE keeps the original date range; EXTEND expands it
    :return: lagged timeseries
    """
    # Extract .w if a Window was passed
    obs = getattr(obs, "w", obs)

    if isinstance(obs, str):
        if re.search("[bB]", obs):
            raise QtkValueError(
                f"Business day offset '{obs}' is not supported. "
                "Use an integer offset for business-day-indexed series."
            )
        return _lag_by_tenor(x, obs, mode)

    if not isinstance(obs, int):
        raise QtkValueError(f"obs must be an int or tenor string, got {type(obs)}")

    if obs == 0:
        return x

    if x.is_empty():
        return x

    if mode == LagMode.EXTEND:
        return _lag_int_extend(x, obs)
    else:
        return x.with_columns(pl.col("value").shift(obs)).drop_nulls()


def _lag_by_tenor(x: pl.DataFrame, tenor: str, mode: LagMode) -> pl.DataFrame:
    """Shift each date in x forward by `tenor`."""
    if x.is_empty():
        return x
    dur = _to_polars_duration(tenor)
    shifted = x.with_columns(pl.col("date").dt.offset_by(dur).alias("date"))
    if mode == LagMode.TRUNCATE:
        last_date = x["date"][-1]
        shifted = shifted.filter(pl.col("date") <= last_date)
    return shifted


def _lag_int_extend(x: pl.DataFrame, obs: int) -> pl.DataFrame:
    """Extend the date index by abs(obs) daily rows then shift values."""
    if obs > 0:
        last_date = x["date"][-1]
        new_dates = [last_date + dt.timedelta(days=i + 1) for i in range(obs)]
        ext = pl.DataFrame({"date": new_dates, "value": [None] * obs}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )
        extended = pl.concat([x, ext])
    else:
        first_date = x["date"][0]
        new_dates = sorted(
            [first_date + dt.timedelta(days=obs + i) for i in range(-obs)]
        )
        ext = pl.DataFrame({"date": new_dates, "value": [None] * (-obs)}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )
        extended = pl.concat([ext, x])

    return extended.with_columns(pl.col("value").shift(obs)).drop_nulls()
