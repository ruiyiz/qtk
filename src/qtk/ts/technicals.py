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
#   - seasonally_adjusted and trend convert to pandas internally for statsmodels
#   All mathematical logic is unchanged from the original.
# -----------------------------------------------------------------------

"""
Technicals library contains technical analysis functions including moving averages,
volatility indicators, and seasonal decomposition.
"""

from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import statsmodels.tsa.seasonal

import polars as pl

from qtk.errors import QtkValueError
from qtk.ts.analysis import diff
from qtk.ts.econometrics import annualize, returns
from qtk.ts.helper import (
    Window,
    _to_timedelta,
    apply_ramp,
    normalize_window,
)
from qtk.ts.statistics import exponential_std, mean, std

__all__ = [
    "Seasonality",
    "SeasonalModel",
    "Frequency",
    "moving_average",
    "bollinger_bands",
    "smoothed_moving_average",
    "relative_strength_index",
    "exponential_moving_average",
    "macd",
    "exponential_volatility",
    "exponential_spread_volatility",
    "seasonally_adjusted",
    "trend",
]


class Seasonality(Enum):
    MONTH = "month"
    QUARTER = "quarter"


class SeasonalModel(Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class Frequency(Enum):
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


def moving_average(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Simple moving average over specified window.

    :param x: timeseries
    :param w: window size
    :return: rolling mean series
    """
    w = normalize_window(x, w)
    return apply_ramp(mean(x, Window(w.w, 0)), w)


def bollinger_bands(
    x: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
    k: float = 2.0,
) -> pl.DataFrame:
    """
    Bollinger bands: moving average +/- k standard deviations.

    :param x: timeseries
    :param w: window size
    :param k: band width in standard deviations (default 2)
    :return: DataFrame with columns date, lower, upper
    """
    w = normalize_window(x, w)
    avg = moving_average(x, w)
    sigma = std(x, w)

    return avg.join(sigma, on="date", how="inner", suffix="_std").with_columns(
        (pl.col("value") - k * pl.col("value_std")).alias("lower"),
        (pl.col("value") + k * pl.col("value_std")).alias("upper"),
    ).select(["date", "lower", "upper"])


def smoothed_moving_average(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Smoothed (modified) moving average.

    :param x: timeseries
    :param w: window size
    :return: smoothed moving average series
    """
    w = normalize_window(x, w)
    means = apply_ramp(mean(x, Window(w.w, 0)), w)
    if means.is_empty():
        return pl.DataFrame({"date": [], "value": []}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )

    initial_ma = means["value"][0]

    if (isinstance(w.r, int) and w.r > 0) or isinstance(w.r, str):
        x = apply_ramp(x, w)

    dates = x["date"].to_list()
    values = x["value"].to_list()
    result = [initial_ma]

    for i in range(1, len(values)):
        if isinstance(w.w, int):
            n = w.w
        else:
            td = _to_timedelta(w.w)
            n = sum(1 for d in dates[: i + 1] if dates[i] - d < td)
        result.append(((n - 1) * result[-1] + values[i]) / n)

    return pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def relative_strength_index(
    x: pl.DataFrame, w: Union[Window, int, str] = 14
) -> pl.DataFrame:
    """
    Relative Strength Index (RSI).

    :param x: timeseries of prices
    :param w: window size (default 14)
    :return: RSI series (0-100)
    """
    w = normalize_window(x, w)
    one_period_change = diff(x, 1)

    gains = one_period_change.with_columns(
        pl.col("value").clip(lower_bound=0.0).alias("value")
    )
    losses = one_period_change.with_columns(
        (-pl.col("value").clip(upper_bound=0.0)).alias("value")
    )

    ma_gains = smoothed_moving_average(gains, w)
    ma_losses = smoothed_moving_average(losses, w)

    joined = ma_gains.join(ma_losses, on="date", how="inner", suffix="_losses")
    return joined.with_columns(
        pl.when(pl.col("value_losses") == 0)
        .then(pl.lit(100.0))
        .otherwise(100.0 - 100.0 / (1.0 + pl.col("value") / pl.col("value_losses")))
        .alias("value")
    ).select(["date", "value"])


def exponential_moving_average(x: pl.DataFrame, beta: float = 0.75) -> pl.DataFrame:
    """
    Exponentially weighted moving average: Y_t = beta * Y_{t-1} + (1-beta) * X_t.

    :param x: timeseries
    :param beta: decay factor (weight on previous average), in [0, 1)
    :return: EWM average series
    """
    return x.with_columns(
        pl.col("value").ewm_mean(alpha=1 - beta, adjust=False).alias("value")
    )


def macd(
    x: pl.DataFrame, m: int = 12, n: int = 26, s: int = 1
) -> pl.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    :param x: timeseries
    :param m: short EMA period (default 12)
    :param n: long EMA period (default 26)
    :param s: signal smoothing period (default 1, i.e. no smoothing)
    :return: MACD series
    """
    from qtk.ts.algebra import subtract

    alpha_m = 2.0 / (m + 1)
    alpha_n = 2.0 / (n + 1)
    alpha_s = 2.0 / (s + 1)

    ema_m = x.with_columns(pl.col("value").ewm_mean(alpha=alpha_m, adjust=False).alias("value"))
    ema_n = x.with_columns(pl.col("value").ewm_mean(alpha=alpha_n, adjust=False).alias("value"))
    diff_mn = subtract(ema_m, ema_n)
    return diff_mn.with_columns(
        pl.col("value").ewm_mean(alpha=alpha_s, adjust=False).alias("value")
    )


def exponential_volatility(x: pl.DataFrame, beta: float = 0.75) -> pl.DataFrame:
    """
    Annualized exponentially weighted volatility of returns.

    :param x: timeseries of prices
    :param beta: decay factor
    :return: annualized EWM volatility series (in percent)
    """
    return annualize(exponential_std(returns(x), beta)).with_columns(
        (pl.col("value") * 100.0).alias("value")
    )


def exponential_spread_volatility(x: pl.DataFrame, beta: float = 0.75) -> pl.DataFrame:
    """
    Annualized exponentially weighted volatility of spread (first differences).

    :param x: timeseries of prices
    :param beta: decay factor
    :return: annualized EWM spread volatility series
    """
    return annualize(exponential_std(diff(x, 1), beta))


# ---------------------------------------------------------------------------
# Seasonal decomposition (uses statsmodels internally)
# ---------------------------------------------------------------------------


def _to_pandas_series(x: pl.DataFrame) -> pd.Series:
    """Convert polars date/value DataFrame to pandas Series with DatetimeIndex."""
    dates = pd.to_datetime(x["date"].to_list())
    values = x["value"].to_numpy(allow_copy=True)
    return pd.Series(values, index=dates)


def _from_pandas_series(s: pd.Series) -> pl.DataFrame:
    """Convert a pandas Series with DatetimeIndex back to polars date/value DataFrame."""
    dates = [d.date() for d in s.index]
    return pl.DataFrame({"date": dates, "value": s.to_numpy()}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def _freq_to_period(s: pd.Series, freq: Frequency = Frequency.YEAR):
    """Infer decomposition period from pandas series frequency."""
    pfreq = getattr(getattr(s, "index", None), "inferred_freq", None)
    pfreq = "MS" if pfreq in ("ME", "M") else pfreq
    pfreq = "QS" if pfreq in ("QE-DEC", "QE") else pfreq
    period = None if pfreq is None else statsmodels.tsa.seasonal.freq_to_period(pfreq)

    if period in [7, None]:  # daily
        s = s.asfreq("D", method="ffill")
        if freq == Frequency.YEAR:
            return s, 365
        elif freq == Frequency.QUARTER:
            return s, 91
        elif freq == Frequency.MONTH:
            return s, 30
        else:
            return s, 7
    elif period == 5:  # business day
        if freq == Frequency.YEAR:
            return s.asfreq("D", method="ffill"), 365
        elif freq == Frequency.QUARTER:
            return s.asfreq("D", method="ffill"), 91
        elif freq == Frequency.MONTH:
            return s.asfreq("D", method="ffill"), 30
        else:
            return s.asfreq("B", method="ffill"), 5
    elif period == 52:  # weekly
        s = s.asfreq("W", method="ffill")
        if freq == Frequency.YEAR:
            return s, period
        elif freq == Frequency.QUARTER:
            return s, 13
        elif freq == Frequency.MONTH:
            return s, 4
        else:
            raise QtkValueError(f"Frequency {freq.value} not compatible with weekly series.")
    elif period == 12:  # monthly
        s = s.asfreq("ME", method="ffill")
        if freq == Frequency.YEAR:
            return s, period
        elif freq == Frequency.QUARTER:
            return s, 3
        else:
            raise QtkValueError(f"Frequency {freq.value} not compatible with monthly series.")
    return s, period


def _seasonal_decompose(
    x: pl.DataFrame,
    method: SeasonalModel = SeasonalModel.ADDITIVE,
    freq: Frequency = Frequency.YEAR,
):
    s = _to_pandas_series(x)
    s, period = _freq_to_period(s, freq)
    if s.shape[0] < 2 * period:
        raise QtkValueError(
            f"Series must have two complete cycles to be analyzed. "
            f"Series has only {s.shape[0]} data points."
        )
    return statsmodels.tsa.seasonal.seasonal_decompose(s, period=period, model=method.value)


def seasonally_adjusted(
    x: pl.DataFrame,
    method: SeasonalModel = SeasonalModel.ADDITIVE,
    freq: Frequency = Frequency.YEAR,
) -> pl.DataFrame:
    """
    Seasonally adjusted series (trend + residual).

    :param x: timeseries with at least two full cycles of data
    :param method: ADDITIVE or MULTIPLICATIVE seasonal model
    :param freq: period of the seasonal cycle (YEAR, QUARTER, MONTH, WEEK)
    :return: de-seasonalized series
    """
    decomp = _seasonal_decompose(x, method, freq)
    if method == SeasonalModel.ADDITIVE:
        result = decomp.trend + decomp.resid
    else:
        result = decomp.trend * decomp.resid
    return _from_pandas_series(result.dropna())


def trend(
    x: pl.DataFrame,
    method: SeasonalModel = SeasonalModel.ADDITIVE,
    freq: Frequency = Frequency.YEAR,
) -> pl.DataFrame:
    """
    Trend component of seasonal decomposition.

    :param x: timeseries with at least two full cycles of data
    :param method: ADDITIVE or MULTIPLICATIVE seasonal model
    :param freq: period of the seasonal cycle (YEAR, QUARTER, MONTH, WEEK)
    :return: trend component series
    """
    decomp = _seasonal_decompose(x, method, freq)
    return _from_pandas_series(decomp.trend.dropna())
