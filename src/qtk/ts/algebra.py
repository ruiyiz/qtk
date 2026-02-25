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
#   All mathematical logic is unchanged from the original.
# -----------------------------------------------------------------------

"""
Algebra library contains basic numerical and algebraic operations, including addition, division, multiplication,
division and other functions on timeseries
"""

import datetime as dt
import math
from enum import Enum
from functools import reduce
from numbers import Real
from typing import List, Optional, Union

import numpy as np
import polars as pl

from qtk.errors import QtkTypeError, QtkValueError
from qtk.ts.dateops import align
from qtk.ts.helper import Interpolate

__all__ = [
    "FilterOperator",
    "add",
    "subtract",
    "multiply",
    "divide",
    "floordiv",
    "exp",
    "log",
    "power",
    "sqrt",
    "abs_",
    "floor",
    "ceil",
    "filter_",
    "filter_dates",
    "and_",
    "or_",
    "not_",
    "if_",
    "weighted_sum",
    "geometrically_aggregate",
]


class FilterOperator(Enum):
    LESS = "less_than"
    GREATER = "greater_than"
    L_EQUALS = "l_equals"
    G_EQUALS = "g_equals"
    EQUALS = "equals"
    N_EQUALS = "not_equals"


def add(
    x: Union[pl.DataFrame, Real],
    y: Union[pl.DataFrame, Real],
    method: Interpolate = Interpolate.STEP,
) -> Union[pl.DataFrame, Real]:
    """
    Add two series or scalars

    :param x: timeseries or scalar
    :param y: timeseries or scalar
    :param method: interpolation method (default: step). Only used when both x and y are timeseries
    :return: timeseries of x + y or sum of the given real numbers

    **Usage**

    Add two series or scalar variables with the given interpolation method

    :math:`R_t =  X_t + Y_t`

    Alignment operators:

    =========   ========================================================================
    Method      Behavior
    =========   ========================================================================
    intersect   Resultant series only has values on the intersection of dates. Values
                for dates present in only one series will be ignored
    nan         Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as nan in the
                other series, and therefore in the resultant series
    zero        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as zero in the
                other series
    step        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be interpolated via step
                function in the other series
    time        Resultant series have values on the union of dates / times. Missing
                values surrounded by valid values will be interpolated given length of
                interval. Input series must use DateTimeIndex.
    =========   ========================================================================

    **Examples**

    Add two series:

    >>> a = generate_series(100)
    >>> b = generate_series(100)
    >>> add(a, b, Interpolate.STEP)

    **See also**

    :func:`subtract`
    """
    if isinstance(x, Real) and isinstance(y, Real):
        return x + y
    x_align, y_align = align(x, y, method)
    return x_align.with_columns((pl.col("value") + y_align["value"]).alias("value"))


def subtract(
    x: Union[pl.DataFrame, Real],
    y: Union[pl.DataFrame, Real],
    method: Interpolate = Interpolate.STEP,
) -> Union[pl.DataFrame, Real]:
    """
    Add two series or scalars

    :param x: timeseries or scalar
    :param y: timeseries or scalar
    :param method: index alignment operator (default: intersect). Only used when both x and y are timeseries
    :return: timeseries of x - y or difference between the given real numbers

    **Usage**

    Subtracts one series or scalar from another applying the given interpolation method

    :math:`R_t =  X_t - Y_t`

    Alignment operators:

    =========   ========================================================================
    Method      Behavior
    =========   ========================================================================
    intersect   Resultant series only has values on the intersection of dates.
                Values for dates present in only one series will be ignored
    nan         Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as nan in the
                other series, and therefore in the resultant series
    zero        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as zero in the
                other series
    step        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be interpolated via step
                function in the other series
    time        Resultant series have values on the union of dates / times. Missing
                values surrounded by valid values will be interpolated given length of
                interval. Input series must use DateTimeIndex.
    =========   ========================================================================

    **Examples**

    Subtract one series from another:

    >>> a = generate_series(100)
    >>> b = generate_series(100)
    >>> subtract(a, b, Interpolate.STEP)

    **See also**

    :func:`add`
    """
    if isinstance(x, Real) and isinstance(y, Real):
        return x - y
    x_align, y_align = align(x, y, method)
    return x_align.with_columns((pl.col("value") - y_align["value"]).alias("value"))


def multiply(
    x: Union[pl.DataFrame, Real],
    y: Union[pl.DataFrame, Real],
    method: Interpolate = Interpolate.STEP,
) -> Union[pl.DataFrame, Real]:
    """
    Multiply two series or scalars

    :param x: timeseries or scalar
    :param y: timeseries or scalar
    :param method: interpolation method (default: step). Only used when both x and y are timeseries
    :return: timeseries of x * y or product of the given real numbers

    **Usage**

    Multiply two series or scalar variables applying the given interpolation method

    :math:`R_t =  X_t \\times Y_t`

    Alignment operators:

    =========   ========================================================================
    Method      Behavior
    =========   ========================================================================
    intersect   Resultant series only has values on the intersection of dates. Values
                for dates present in only one series will be ignored
    nan         Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as nan in the
                other series, and therefore in the resultant series
    zero        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as zero in the
                other series
    step        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be interpolated via step
                function in the other series
    time        Resultant series have values on the union of dates / times. Missing
                values surrounded by valid values will be interpolated given length of
                interval. Input series must use DateTimeIndex.
    =========   ========================================================================

    **Examples**

    Multiply two series:

    >>> a = generate_series(100)
    >>> b = generate_series(100)
    >>> multiply(a, b, Interpolate.STEP)

    **See also**

    :func:`divide`
    """
    if isinstance(x, Real) and isinstance(y, Real):
        return x * y
    x_align, y_align = align(x, y, method)
    return x_align.with_columns((pl.col("value") * y_align["value"]).alias("value"))


def divide(
    x: Union[pl.DataFrame, Real],
    y: Union[pl.DataFrame, Real],
    method: Interpolate = Interpolate.STEP,
) -> Union[pl.DataFrame, Real]:
    """
    Divide two series or scalars

    :param x: timeseries or scalar
    :param y: timeseries or scalar
    :param method: interpolation method (default: step). Only used when both x and y are timeseries
    :return: timeseries of x / y or quotient of the given real numbers

    **Usage**

    Divide two series or scalar variables applying the given interpolation method

    :math:`R_t =  X_t / Y_t`

    Alignment operators:

    =========   ========================================================================
    Method      Behavior
    =========   ========================================================================
    intersect   Resultant series only has values on the intersection of dates.
                Values for dates present in only one series will be ignored
    nan         Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as nan in the
                other series, and therefore in the resultant series
    zero        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as zero in the
                other series
    step        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be interpolated via step
                function in the other series
    time        Resultant series have values on the union of dates / times. Missing
                values surrounded by valid values will be interpolated given length of
                interval. Input series must use DateTimeIndex.
    =========   ========================================================================

    **Examples**

    Divide two series:

    >>> a = generate_series(100)
    >>> b = generate_series(100)
    >>> divide(a, b, Interpolate.STEP)

    **See also**

    :func:`multiply`
    """
    if isinstance(x, Real) and isinstance(y, Real):
        return x / y
    x_align, y_align = align(x, y, method)
    return x_align.with_columns((pl.col("value") / y_align["value"]).alias("value"))


def floordiv(
    x: Union[pl.DataFrame, Real],
    y: Union[pl.DataFrame, Real],
    method: Interpolate = Interpolate.STEP,
) -> Union[pl.DataFrame, Real]:
    """
    Floor divide two series or scalars

    :param x: timeseries or scalar
    :param y: timeseries or scalar
    :param method: interpolation method (default: step). Only used for operating two series
    :return: timeseries of x // y or quotient of the floor division of the given real numbers

    **Usage**

    Divide two series or scalar variables applying the given interpolation method

    :math:`R_t =  X_t / Y_t`

    Alignment operators:

    =========   ========================================================================
    Method      Behavior
    =========   ========================================================================
    intersect   Resultant series only has values on the intersection of dates.
                Values for dates present in only one series will be ignored
    nan         Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as nan in the
                other series, and therefore in the resultant series
    zero        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be treated as zero in the
                other series
    step        Resultant series has values on the union of dates in both series. Values
                for dates only available in one series will be interpolated via step
                function in the other series
    =========   ========================================================================

    **Examples**

    Floor divide two series:

    >>> a = generate_series(100)
    >>> b = generate_series(100)
    >>> floordiv(a, b, Interpolate.STEP)

    **See also**

    :func:`divide`
    """
    if isinstance(x, Real) and isinstance(y, Real):
        return x // y
    x_align, y_align = align(x, y, method)
    return x_align.with_columns((pl.col("value") // y_align["value"]).alias("value"))


def exp(x: pl.DataFrame) -> pl.DataFrame:
    """
    Exponential of series

    :param x: timeseries
    :return: exponential of each element

    **Usage**

    For each element in the series, :math:`X_t`, raise :math:`e` (Euler's number) to the power of :math:`X_t`.
    Euler's number is the base of the natural logarithm, :math:`ln`.

    :math:`R_t = e^{X_t}`

    **Examples**

    Raise :math:`e` to the power :math:`1`. Returns Euler's number, approximately 2.71828

    >>> exp(1)

    **See also**

    :func:`log`

    """
    return x.with_columns(pl.col("value").exp())


def log(x: pl.DataFrame) -> pl.DataFrame:
    """
    Natural logarithm of series

    :param x: timeseries
    :return: series with exponential of each element

    **Usage**

    For each element in the series, :math:`X_t`, return the natural logarithm :math:`ln` of :math:`X_t`
    The natural logarithm is the logarithm in base :math:`e`.

    :math:`R_t = log(X_t)`

    This function is the inverse of the exponential function.

    More information on `logarithms <https://en.wikipedia.org/wiki/Logarithm>`_

    **Examples**

    Take natural logarithm of 3

    >>> log(3)

    **See also**

    :func:`exp`

    """
    return x.with_columns(pl.col("value").log(base=math.e))


def power(x: pl.DataFrame, y: float = 1) -> pl.DataFrame:
    """
    Raise each element in series to power

    :param x: timeseries
    :param y: value
    :return: date-based time series of square roots

    **Usage**

    Raise each value in time series :math:`X_t` to the power :math:`y`:

    :math:`R_t = X_t^{y}`

    **Examples**

    Generate price series and raise each value to the power 2:

    >>> prices = generate_series(100)
    >>> power(prices, 2)

    **See also**

    :func:`sqrt`

    """
    return x.with_columns(pl.col("value").pow(y))


def sqrt(x: Union[Real, pl.DataFrame]) -> Union[Real, pl.DataFrame]:
    """
    Square root of (a) each element in a series or (b) a real number

    :param x: date-based time series of prices or real number
    :return: date-based time series of square roots or square root of given number

    **Usage**

    Return the square root of each value in time series :math:`X_t`:

    :math:`R_t = \\sqrt{X_t}`

    **Examples**

    Generate price series and take square root of each value:

    >>> prices = generate_series(100)
    >>> sqrt(prices)

    **See also**

    :func:`pow`

    """
    if isinstance(x, pl.DataFrame):
        return x.with_columns(pl.col("value").sqrt())
    result = math.sqrt(x)
    return round(result) if round(result) == result else result


def abs_(x: pl.DataFrame) -> pl.DataFrame:
    """
    Absolute value of each element in series

    :param x: date-based time series of prices
    :return: date-based time series of absolute value

    **Usage**

    Return the absolute value of :math:`X`. For each value in time series :math:`X_t`, return :math:`X_t` if :math:`X_t`
    is greater than or equal to 0; otherwise return :math:`-X_t`:

    :math:`R_t = |X_t|`

    Equivalent to :math:`R_t = sqrt{X_t^2}`

    **Examples**

    Generate price series and take absolute value of :math:`X_t-100`

    >>> prices = generate_series(100) - 100
    >>> abs_(prices)

    **See also**

    :func:`exp` :func:`sqrt`

    """
    return x.with_columns(pl.col("value").abs())


def floor(x: pl.DataFrame, value: float = 0) -> pl.DataFrame:
    """
    Floor series at minimum value

    :param x: date-based time series of prices
    :param value: minimum value
    :return: date-based time series of maximum value

    **Usage**

    Returns series where all values are greater than or equal to the minimum value.

    :math:`R_t = max(X_t, value)`

    See `Floor and Ceil functions <https://en.wikipedia.org/wiki/Floor_and_ceiling_functions>`_ for more details

    **Examples**

    Generate price series and floor all values at 100

    >>> prices = generate_series(100)
    >>> floor(prices, 100)

    **See also**

    :func:`ceil`

    """
    return x.with_columns(
        pl.when(pl.col("value") < value).then(value).otherwise(pl.col("value")).alias("value")
    )


def ceil(x: pl.DataFrame, value: float = 0) -> pl.DataFrame:
    """
    Cap series at maximum value

    :param x: date-based time series of prices
    :param value: maximum value
    :return: date-based time series of maximum value

    **Usage**

    Returns series where all values are less than or equal to the maximum value.

    :math:`R_t = min(X_t, value)`

    See `Floor and Ceil functions <https://en.wikipedia.org/wiki/Floor_and_ceiling_functions>`_ for more details

    **Examples**

    Generate price series and floor all values at 100

    >>> prices = generate_series(100)
    >>> floor(prices, 100)

    **See also**

    :func:`floor`

    """
    return x.with_columns(
        pl.when(pl.col("value") > value).then(value).otherwise(pl.col("value")).alias("value")
    )


def filter_(
    x: pl.DataFrame,
    operator: Optional[FilterOperator] = None,
    value: Optional[Real] = None,
) -> pl.DataFrame:
    """
    Removes values where comparison with the operator and value combination results in true, defaults to removing
    missing values from the series

    :param x: timeseries
    :param operator: FilterOperator describing logic for value removal, e.g 'less_than'
    :param value: number indicating value(s) to remove from the series
    :return: timeseries with specified values removed


    **Usage**

    Remove each value determined by operator and value from timeseries where that expression yields true

    **Examples**

    Remove 0 from time series

    >>> prices = generate_series(100)
    >>> filter_(prices, FilterOperator.EQUALS, 0)

    Remove positive numbers from time series

    >>> prices = generate_series(100)
    >>> filter_(prices, FilterOperator.GREATER, 0)

    Remove missing values from time series

    >>> prices = generate_series(100)
    >>> filter_(prices)

    """
    if value is None and operator is None:
        return x.drop_nulls()
    elif value is None:
        raise QtkValueError("No value is specified for the operator")
    else:
        if operator == FilterOperator.EQUALS:
            return x.filter(pl.col("value") != value)
        elif operator == FilterOperator.GREATER:
            return x.filter(pl.col("value") <= value)
        elif operator == FilterOperator.LESS:
            return x.filter(pl.col("value") >= value)
        elif operator == FilterOperator.L_EQUALS:
            return x.filter(pl.col("value") > value)
        elif operator == FilterOperator.G_EQUALS:
            return x.filter(pl.col("value") < value)
        elif operator == FilterOperator.N_EQUALS:
            return x.filter(pl.col("value") == value)
        else:
            raise QtkValueError("Unexpected operator: " + str(operator))


def filter_dates(
    x: pl.DataFrame,
    operator: Optional[FilterOperator] = None,
    dates: Union[List[dt.date], "dt.date", None] = None,
) -> pl.DataFrame:
    """
    Removes dates where comparison with the operator and dates combination results in true, defaults to removing
    missing values from the series

    :param x: timeseries
    :param operator: FilterOperator describing logic for date removal, e.g 'less_than'
    :param dates: date or list of dates to remove from the series
    :return: timeseries with specified dates removed


    **Usage**

    Remove each date determined by operator and date from timeseries where that expression yields true

    **Examples**

    Remove today from time series

    >>> prices = generate_series(100)
    >>> filter_dates(prices, FilterOperator.EQUALS, date.today())

    Remove dates before today from time series

    >>> prices = generate_series(100)
    >>> filter_dates(prices, FilterOperator.LESS, date.today())

    """
    import datetime as dt_mod

    if dates is None and operator is None:
        return x.drop_nulls()
    elif dates is None:
        raise QtkValueError("No date is specified for the operator")
    elif isinstance(dates, list) and operator not in [FilterOperator.EQUALS, FilterOperator.N_EQUALS]:
        raise QtkValueError("Operator does not work for list of dates")
    else:
        if operator == FilterOperator.EQUALS:
            date_list = dates if isinstance(dates, list) else [dates]
            return x.filter(~pl.col("date").is_in(date_list))
        elif operator == FilterOperator.N_EQUALS:
            date_list = dates if isinstance(dates, list) else [dates]
            return x.filter(pl.col("date").is_in(date_list))
        elif operator == FilterOperator.GREATER:
            return x.filter(pl.col("date") <= dates)
        elif operator == FilterOperator.LESS:
            return x.filter(pl.col("date") >= dates)
        elif operator == FilterOperator.L_EQUALS:
            return x.filter(pl.col("date") > dates)
        elif operator == FilterOperator.G_EQUALS:
            return x.filter(pl.col("date") < dates)
        else:
            raise QtkValueError("Unexpected operator: " + str(operator))


def _check_boolean_series(series: List[pl.DataFrame]):
    if not 2 <= len(series) <= 100:
        raise QtkValueError("expected between 2 and 100 arguments")
    for s in series:
        if not isinstance(s, pl.DataFrame):
            raise QtkTypeError("all arguments must be DataFrames")
        vals = s["value"].drop_nulls().to_list()
        if not all(v in (0, 1) for v in vals):
            raise QtkValueError(f"cannot perform operation on series with value(s) other than 1 and 0: {vals}")


def and_(*series: pl.DataFrame) -> pl.DataFrame:
    """
    Logical "and" of two or more boolean series.

    :param series: input series
    :return: result series (of numeric type, with booleans represented as 1s and 0s)
    """
    series = list(series)
    _check_boolean_series(series)
    # align all on union of dates
    base = series[0]
    for s in series[1:]:
        joined = base.join(s, on="date", how="full", suffix="_r", coalesce=True).sort("date")
        base = joined.with_columns(
            (pl.col("value").fill_null(0) + pl.col("value_r").fill_null(0)).alias("value")
        ).select("date", "value")
    return base.with_columns(
        pl.when(pl.col("value") == len(series)).then(1).otherwise(0).alias("value")
    )


def or_(*series: pl.DataFrame) -> pl.DataFrame:
    """
    Logical "or" of two or more boolean series.

    :param series: input series
    :return: result series (of numeric type, with booleans represented as 1s and 0s)
    """
    series = list(series)
    _check_boolean_series(series)
    base = series[0]
    for s in series[1:]:
        joined = base.join(s, on="date", how="full", suffix="_r", coalesce=True).sort("date")
        base = joined.with_columns(
            (pl.col("value").fill_null(0) + pl.col("value_r").fill_null(0)).alias("value")
        ).select("date", "value")
    return base.with_columns(
        pl.when(pl.col("value") > 0).then(1).otherwise(0).alias("value")
    )


def not_(series: pl.DataFrame) -> pl.DataFrame:
    """
    Logical negation of a single boolean series.

    :param series: single input series
    :return: result series (of numeric type, with booleans represented as 1s and 0s)
    """
    vals = series["value"].drop_nulls().to_list()
    if not all(v in (0, 1) for v in vals):
        raise QtkValueError(f"cannot negate series with value(s) other than 1 and 0: {vals}")
    return series.with_columns(
        pl.when(pl.col("value") == 0).then(1).otherwise(0).alias("value")
    )


def if_(
    flags: pl.DataFrame,
    x: Union[pl.DataFrame, float],
    y: Union[pl.DataFrame, float],
) -> pl.DataFrame:
    """
    Returns a series s. For i in the index of flags, s[i] = x[i] if flags[i] == 1 else y[i].

    :param flags: series of 1s and 0s
    :param x: values to use when flag is 1
    :param y: values to use when flag is 0
    :return: result series

    **Usage**

    Returns a series based off the given conditional series. If the condition is true it shows the first series's value
    else it shows the second series's value.

    **PlotTool Example**

    if(SPX.spot() > 4000, SPX.spot(), GSTHHVIP.spot())

    The above expression would show SPX.spot() if the spot price is above 4000 else it shows GSTHHVIP.spot().

    """
    flag_vals = flags["value"].drop_nulls().to_list()
    if not all(v in (0, 1) for v in flag_vals):
        raise QtkValueError(f"cannot perform 'if' on series with value(s) other than 1 and 0: {flag_vals}")

    def _ensure_df(s) -> pl.DataFrame:
        if isinstance(s, (float, int)):
            return flags.with_columns(pl.lit(float(s)).alias("value"))
        elif isinstance(s, pl.DataFrame):
            return s
        else:
            raise QtkTypeError("expected a number or series")

    x_df = _ensure_df(x)
    y_df = _ensure_df(y)

    x_part = x_df.join(flags, on="date", how="inner", suffix="_flag").filter(
        pl.col("value_flag") == 1
    ).select("date", "value")
    y_part = y_df.join(flags, on="date", how="inner", suffix="_flag").filter(
        pl.col("value_flag") == 0
    ).select("date", "value")
    return pl.concat([x_part, y_part]).sort("date")


def weighted_sum(series: List[pl.DataFrame], weights: list) -> pl.DataFrame:
    """
    Calculate a weighted sum.

    :param series: list of time series
    :param weights: list of weights
    :return: time series of weighted average

    **Usage**

    Calculate a weighted sum e.g. for a basket.

    **Examples**

    Generate price series and get a sum (weights 70%/30%).

    >>> prices1 = generate_series(100)
    >>> prices2 = generate_series(100)
    >>> mybasket = weighted_sum([prices1, prices2], [0.7, 0.3])

    **See also**

    :func:`basket`
    """
    if not all(isinstance(x, pl.DataFrame) for x in series):
        raise QtkTypeError("expected a list of time series")
    if not all(isinstance(w, (float, int)) for w in weights):
        raise QtkTypeError("expected a list of number for weights")
    if len(weights) != len(series):
        raise QtkValueError("must have one weight for each time series")

    # find intersection of all date sets
    dates_sets = [set(s["date"].to_list()) for s in series]
    common_dates = sorted(reduce(lambda a, b: a & b, dates_sets))

    weighted = pl.DataFrame({"date": common_dates, "value": [0.0] * len(common_dates)})

    for s, w in zip(series, weights):
        filtered = s.filter(pl.col("date").is_in(common_dates)).sort("date")
        weighted = weighted.with_columns(
            (pl.col("value") + filtered["value"] * w).alias("value")
        )
    return weighted


def geometrically_aggregate(series: pl.DataFrame) -> pl.DataFrame:
    """
    Geometrically aggregate a series.

    :param series: list of time series
    :return: time series of geometrically aggregated results

    **Usage**

    Used to aggregate daily returns when expressed as weights
    """
    return series.with_columns(
        ((pl.col("value") + 1).cum_prod() - 1).alias("value")
    )
