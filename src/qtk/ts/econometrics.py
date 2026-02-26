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
#   - Removed Marquee-only functions: excess_returns_, sharpe_ratio, SharpeAssets
#   - excess_returns() retains only the float benchmark path
#   All mathematical logic is unchanged from the original.
# -----------------------------------------------------------------------

"""
Econometrics timeseries library contains standard economic and time series analytics
operations including returns, prices, volatility, correlation, beta, and drawdown.
"""

import math
from enum import IntEnum, Enum
from typing import List, Optional, Union

import numpy as np
import polars as pl

from qtk.errors import QtkTypeError, QtkValueError
from qtk.date_utils import DayCountConvention, day_count_fraction
from qtk.ts.analysis import LagMode, lag
from qtk.ts.dateops import align, interpolate
from qtk.ts.helper import (
    CurveType,
    Interpolate,
    Returns,
    SeriesType,
    Window,
    _to_timedelta,
    apply_ramp,
    normalize_window,
)
from qtk.ts.statistics import MeanType, mean, std

__all__ = [
    "AnnualizationFactor",
    "RiskFreeRateCurrency",
    "excess_returns_pure",
    "excess_returns",
    "get_ratio_pure",
    "returns",
    "prices",
    "index",
    "change",
    "annualize",
    "volatility",
    "vol_swap_volatility",
    "correlation",
    "corr_swap_correlation",
    "beta",
    "max_drawdown",
    "alpha",
    "bull_beta",
    "bear_beta",
    "timing_ratio",
    "tracking_error",
    "active_premium",
    "up_capture",
    "down_capture",
    "up_capture_number",
    "down_capture_number",
    "up_capture_percent",
    "down_capture_percent",
    "persistence_score",
    "systematic_risk",
    "specific_risk",
]


class AnnualizationFactor(IntEnum):
    DAILY = 252
    WEEKLY = 52
    SEMI_MONTHLY = 26
    MONTHLY = 12
    QUARTERLY = 4
    ANNUALLY = 1


class RiskFreeRateCurrency(Enum):
    USD = "USD"
    AUD = "AUD"
    CHF = "CHF"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    SEK = "SEK"


# ---------------------------------------------------------------------------
# Excess returns
# ---------------------------------------------------------------------------


def excess_returns_pure(
    price_series: pl.DataFrame, spot_curve: pl.DataFrame
) -> pl.DataFrame:
    """
    Compute excess returns of a price series relative to a spot rate curve.

    :param price_series: price timeseries
    :param spot_curve: risk-free spot rate curve timeseries
    :return: excess return timeseries
    """
    curve, bench = align(price_series, spot_curve, Interpolate.INTERSECT)
    curve_vals = curve["value"].to_list()
    bench_vals = bench["value"].to_list()
    dates = curve["date"].to_list()

    er = [curve_vals[0]]
    for i in range(1, len(curve_vals)):
        mult = 1 + curve_vals[i] / curve_vals[i - 1] - bench_vals[i] / bench_vals[i - 1]
        er.append(er[-1] * mult)

    return pl.DataFrame({"date": dates, "value": er}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def excess_returns(
    price_series: pl.DataFrame,
    benchmark_or_rate: float,
    *,
    day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_360,
) -> pl.DataFrame:
    """
    Compute excess returns given a flat risk-free rate.

    :param price_series: price timeseries
    :param benchmark_or_rate: flat risk-free rate as a float (annualized)
    :param day_count_convention: day count convention for accrual
    :return: excess return timeseries

    Note: the Asset/Currency overloads from the original gs-quant require
    Marquee API access and are not available in qtk. Pass a plain float
    for benchmark_or_rate.
    """
    if not isinstance(benchmark_or_rate, float):
        raise QtkValueError(
            "This overload requires GS Marquee access. "
            "Pass a plain float for benchmark_or_rate instead."
        )

    dates = price_series["date"].to_list()
    vals = price_series["value"].to_list()

    er = [vals[0]]
    for j in range(1, len(vals)):
        fraction = day_count_fraction(dates[j - 1], dates[j], day_count_convention)
        er.append(er[-1] + vals[j] - vals[j - 1] * (1 + benchmark_or_rate * fraction))

    return pl.DataFrame({"date": dates, "value": er}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


# ---------------------------------------------------------------------------
# Annualized return helper (used by get_ratio_pure)
# ---------------------------------------------------------------------------


def _annualized_return(
    levels: pl.DataFrame,
    rolling: Union[int, str],
    interpolation_method: Interpolate = Interpolate.NAN,
) -> pl.DataFrame:
    n = levels.height
    dates = levels["date"].to_list()
    vals = levels["value"].to_numpy(allow_copy=True)

    if isinstance(rolling, str):
        td = _to_timedelta(rolling)
        points = [0.0]
        for t in range(1, n):
            cutoff = dates[t] - td
            i_start = None
            for j in range(t - 1, -1, -1):
                if dates[j] <= cutoff:
                    i_start = j
                    break
            if i_start is None:
                points.append(np.nan)
                continue
            day_diff = (dates[t] - dates[i_start]).days
            if day_diff == 0:
                points.append(0.0)
            else:
                points.append(pow(vals[t] / vals[i_start], 365.25 / day_diff) - 1)
    else:
        starting = [0] * rolling
        starting.extend(list(range(1, n - rolling + 1)))
        points = [0.0]
        for idx in range(n - 1):
            t = idx + 1
            i = starting[t]
            day_diff = (dates[t] - dates[i]).days
            if day_diff == 0:
                points.append(np.nan)
            else:
                points.append(pow(vals[t] / vals[i], 365.25 / day_diff) - 1)

    return pl.DataFrame({"date": dates, "value": points}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def get_ratio_pure(
    er: pl.DataFrame,
    w: Union[Window, int, str],
    interpolation_method: Interpolate = Interpolate.NAN,
) -> pl.DataFrame:
    """
    Compute annualized return over annualized volatility (Sharpe-like ratio).

    :param er: excess return timeseries
    :param w: window size
    :param interpolation_method: interpolation for date gaps
    :return: ratio timeseries (ratio * 100)
    """
    w = normalize_window(er, w or None)
    ann_return = _annualized_return(er, w.w, interpolation_method=interpolation_method)

    if isinstance(w.w, str):
        td = _to_timedelta(w.w)
        long_enough = (er["date"][-1] - er["date"][0]) >= td
    else:
        long_enough = w.w < er.height

    if long_enough:
        vol_full = volatility(er, w)
        ann_vol = vol_full.slice(1)
    else:
        ann_vol = volatility(er)

    joined = ann_return.join(ann_vol, on="date", how="inner", suffix="_vol")
    result = joined.with_columns(
        (pl.col("value") / pl.col("value_vol") * 100).alias("value")
    ).select(["date", "value"])
    return apply_ramp(result, w)


# ---------------------------------------------------------------------------
# Core finance functions
# ---------------------------------------------------------------------------


def returns(
    series: pl.DataFrame,
    obs: Union[Window, int, str] = 1,
    type: Returns = Returns.SIMPLE,
) -> pl.DataFrame:
    """
    Calculate returns from a price series.

    :param series: timeseries of prices
    :param obs: lag: integer observations or tenor string e.g. '3d', '1m'
    :param type: SIMPLE, LOGARITHMIC, or ABSOLUTE
    :return: return series
    """
    if series.is_empty():
        return series

    shifted = lag(series, obs, LagMode.TRUNCATE)
    joined = series.join(shifted, on="date", how="inner", suffix="_lag")

    if type == Returns.SIMPLE:
        result = joined.with_columns(
            (pl.col("value") / pl.col("value_lag") - 1).alias("value")
        )
    elif type == Returns.LOGARITHMIC:
        result = joined.with_columns(
            (
                pl.col("value").log(base=math.e) - pl.col("value_lag").log(base=math.e)
            ).alias("value")
        )
    elif type == Returns.ABSOLUTE:
        result = joined.with_columns(
            (pl.col("value") - pl.col("value_lag")).alias("value")
        )
    else:
        raise QtkValueError(
            "Unknown returns type (use simple / logarithmic / absolute)"
        )

    return result.select(["date", "value"])


def prices(
    series: pl.DataFrame,
    initial: float = 1.0,
    type: Returns = Returns.SIMPLE,
) -> pl.DataFrame:
    """
    Compute price levels from a returns series.

    :param series: timeseries of returns
    :param initial: initial price level
    :param type: SIMPLE, LOGARITHMIC, or ABSOLUTE
    :return: price series
    """
    if series.is_empty():
        return series

    if type == Returns.SIMPLE:
        return series.with_columns(
            ((pl.col("value") + 1).cum_prod() * initial).alias("value")
        )
    elif type == Returns.LOGARITHMIC:
        return series.with_columns(
            (pl.col("value").exp().cum_prod() * initial).alias("value")
        )
    elif type == Returns.ABSOLUTE:
        return series.with_columns((pl.col("value").cum_sum() + initial).alias("value"))
    else:
        raise QtkValueError(
            "Unknown returns type (use simple / logarithmic / absolute)"
        )


def index(x: pl.DataFrame, initial: float = 1.0) -> pl.DataFrame:
    """
    Normalize series: Y_t = initial * X_t / X_0.

    :param x: timeseries
    :param initial: initial value
    :return: normalized series
    """
    non_null = x.drop_nulls("value")
    if non_null.is_empty():
        return pl.DataFrame({"date": [], "value": []}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )
    first_val = non_null["value"][0]
    if first_val == 0:
        raise QtkValueError(
            "Divide by zero. Ensure the first value of the series passed to index() is non-zero."
        )
    return x.with_columns((pl.col("value") * initial / first_val).alias("value"))


def change(x: pl.DataFrame) -> pl.DataFrame:
    """
    Compute difference from the first value: Y_t = X_t - X_0.

    :param x: timeseries
    :return: series of changes from initial value
    """
    if x.is_empty():
        return x
    first_val = x.drop_nulls("value")["value"][0]
    return x.with_columns((pl.col("value") - first_val).alias("value"))


def _get_annualization_factor(x: pl.DataFrame) -> int:
    if x.height < 2:
        raise QtkValueError("Series too short to infer annualization factor.")
    dates = x["date"].to_list()
    distances = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    if any(d == 0 for d in distances):
        raise QtkValueError("Multiple data points on the same date.")
    avg = float(np.mean(distances))
    if avg < 2.1:
        return int(AnnualizationFactor.DAILY)
    elif 6 <= avg < 8:
        return int(AnnualizationFactor.WEEKLY)
    elif 14 <= avg < 17:
        return int(AnnualizationFactor.SEMI_MONTHLY)
    elif 25 <= avg < 35:
        return int(AnnualizationFactor.MONTHLY)
    elif 85 <= avg < 97:
        return int(AnnualizationFactor.QUARTERLY)
    elif 360 <= avg < 386:
        return int(AnnualizationFactor.ANNUALLY)
    else:
        raise QtkValueError(
            f"Cannot infer annualization factor, average distance: {avg:.2f} days."
        )


def annualize(x: pl.DataFrame) -> pl.DataFrame:
    """
    Annualize a series by multiplying by sqrt(annualization_factor).

    :param x: timeseries
    :return: annualized series
    """
    factor = _get_annualization_factor(x)
    return x.with_columns((pl.col("value") * math.sqrt(factor)).alias("value"))


def volatility(
    x: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
    returns_type: Optional[Returns] = Returns.SIMPLE,
    annualization_factor: Optional[int] = None,
    assume_zero_mean: bool = False,
) -> pl.DataFrame:
    """
    Realized volatility (annualized, in percent).

    :param x: timeseries of prices or returns
    :param w: rolling window
    :param returns_type: returns conversion; None means x is already returns
    :param annualization_factor: override auto-detected annualization factor
    :param assume_zero_mean: if True, uses zero-mean RMS instead of sample std
    :return: annualized volatility series (e.g. 20.0 = 20%)
    """
    w = normalize_window(x, w)

    if x.is_empty():
        return x

    ret = returns(x, type=returns_type) if returns_type is not None else x
    window = Window(w.w, 0)

    vol = (
        mean(ret, window, MeanType.QUADRATIC) if assume_zero_mean else std(ret, window)
    )

    if annualization_factor is not None:
        ann_vol = vol.with_columns(
            (pl.col("value") * math.sqrt(annualization_factor)).alias("value")
        )
    else:
        ann_vol = annualize(vol)

    result = ann_vol.with_columns((pl.col("value") * 100.0).alias("value"))
    return apply_ramp(result, w)


def vol_swap_volatility(
    prices_series: pl.DataFrame,
    n_days: Union[int, Window, None] = None,
    annualization_factor: int = 252,
    assume_zero_mean: bool = True,
) -> pl.DataFrame:
    """
    Rolling volatility for volatility swap pricing (log returns, zero-mean convention by default).

    :param prices_series: price timeseries
    :param n_days: rolling window in price days (defaults to full series length)
    :param annualization_factor: periods per year (default 252)
    :param assume_zero_mean: if True (default), uses zero-mean RMS (vol swap convention)
    :return: annualized volatility in percent
    """
    if n_days is None:
        n_days = prices_series.height
    if isinstance(n_days, Window):
        if n_days.r != n_days.w - 1:
            raise QtkTypeError(
                "Ramp-up must be window size minus 1, e.g. Window(4, 3)."
            )
        window = n_days
    else:
        window = Window(n_days, n_days - 1)
    return volatility(
        prices_series,
        window,
        Returns.LOGARITHMIC,
        annualization_factor,
        assume_zero_mean,
    )


def correlation(
    x: pl.DataFrame,
    y: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
    type_: SeriesType = SeriesType.PRICES,
    returns_type: Returns = Returns.SIMPLE,
    assume_zero_mean: bool = False,
) -> pl.DataFrame:
    """
    Rolling realized correlation of two series.

    :param x: first price or return series
    :param y: second price or return series
    :param w: rolling window
    :param type_: PRICES or RETURNS input
    :param returns_type: returns conversion method when type_=PRICES
    :param assume_zero_mean: if True, uses zero-mean correlation (correlation swap convention)
    :return: correlation series
    """
    w = normalize_window(x, w)

    if x.is_empty():
        return x

    if type_ == SeriesType.PRICES:
        if isinstance(returns_type, (tuple, list)):
            if len(returns_type) != 2:
                raise QtkValueError('Expected a list of length 2 for "returns_type"')
            returns_type_x, returns_type_y = returns_type
        else:
            returns_type_x = returns_type_y = returns_type
        ret1 = returns(x, type=returns_type_x)
        ret2 = returns(y, type=returns_type_y)
    else:
        ret1, ret2 = x, y

    joined = (
        ret1.join(ret2, on="date", how="inner", suffix="_y").sort("date").drop_nulls()
    )
    x_arr = joined["value"].to_numpy(allow_copy=True)
    y_arr = joined["value_y"].to_numpy(allow_copy=True)
    dates = joined["date"].to_list()
    n = len(x_arr)

    if assume_zero_mean:
        corr_vals = _rolling_zero_mean_corr(x_arr, y_arr, w)
    else:
        corr_vals = _rolling_pearson_corr(x_arr, y_arr, w, dates)

    corr_df = pl.DataFrame({"date": dates, "value": corr_vals}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    result = interpolate(corr_df, x, Interpolate.NAN)
    return apply_ramp(result, w)


def _rolling_pearson_corr(x_arr, y_arr, w: Window, dates) -> np.ndarray:
    n = len(x_arr)
    result = np.full(n, np.nan)
    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi, yi = x_arr[start : i + 1], y_arr[start : i + 1]
            mask = ~(np.isnan(xi) | np.isnan(yi))
            xi, yi = xi[mask], yi[mask]
            if len(xi) >= 2:
                c = np.corrcoef(xi, yi)
                result[i] = c[0, 1]
    else:
        td = _to_timedelta(w.w)
        for i in range(n):
            cutoff = dates[i] - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi, yi = x_arr[idx], y_arr[idx]
            mask = ~(np.isnan(xi) | np.isnan(yi))
            xi, yi = xi[mask], yi[mask]
            if len(xi) >= 2:
                c = np.corrcoef(xi, yi)
                result[i] = c[0, 1]
    return result


def _rolling_zero_mean_corr(x_arr, y_arr, w: Window) -> np.ndarray:
    n = len(x_arr)
    result = np.full(n, np.nan)
    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi, yi = x_arr[start : i + 1], y_arr[start : i + 1]
            mask = ~(np.isnan(xi) | np.isnan(yi))
            xi, yi = xi[mask], yi[mask]
            if len(xi) >= 1:
                dot = np.sum(xi * yi)
                norm = np.sqrt(np.sum(xi**2) * np.sum(yi**2))
                result[i] = dot / norm if norm != 0 else np.nan
    return result


def corr_swap_correlation(
    x: pl.DataFrame,
    y: pl.DataFrame,
    n_days: Union[int, Window, None] = None,
    assume_zero_mean: bool = True,
) -> pl.DataFrame:
    """
    Rolling correlation for correlation swap pricing (log returns, zero-mean by default).

    :param x: first price series
    :param y: second price series
    :param n_days: rolling window in price days (defaults to min series length)
    :param assume_zero_mean: if True (default), uses zero-mean correlation
    :return: correlation series
    """
    if n_days is None:
        n_days = min(x.height, y.height)
    if isinstance(n_days, Window):
        if n_days.r != n_days.w - 1:
            raise QtkTypeError(
                "Ramp-up must be window size minus 1, e.g. Window(4, 3)."
            )
        window = n_days
    else:
        window = Window(n_days - 1, n_days - 2)
    return correlation(
        x, y, window, SeriesType.PRICES, Returns.LOGARITHMIC, assume_zero_mean
    )


def beta(
    x: pl.DataFrame,
    b: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
    prices: bool = True,
) -> pl.DataFrame:
    """
    Rolling beta of a series against a benchmark.

    :param x: timeseries of prices or returns
    :param b: benchmark timeseries of prices or returns
    :param w: rolling window
    :param prices: True if inputs are prices; False if they are already returns
    :return: rolling beta series
    """
    from qtk.ts.statistics import cov, var

    if not isinstance(prices, bool):
        raise QtkTypeError('expected a boolean value for "prices"')

    w = normalize_window(x, w)

    ret_x = returns(x) if prices else x
    ret_b = returns(b) if prices else b

    cov_result = cov(ret_x, ret_b, w)
    var_result = var(ret_b, w)

    joined = cov_result.join(var_result, on="date", how="inner", suffix="_var")
    result = joined.with_columns(
        (pl.col("value") / pl.col("value_var")).alias("value")
    ).select(["date", "value"])

    # Set first 3 values to null as in original (initial values may be extreme)
    if result.height >= 3:
        result = pl.concat(
            [
                result[:3].with_columns(pl.lit(None).cast(pl.Float64).alias("value")),
                result[3:],
            ]
        )

    return apply_ramp(interpolate(result, x, Interpolate.NAN), w)


def max_drawdown(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Rolling maximum peak-to-trough drawdown.

    :param x: timeseries
    :param w: rolling window
    :return: rolling max drawdown (negative values, e.g. -0.2 for -20%)
    """
    w = normalize_window(x, w)

    if isinstance(w.w, int):
        result = (
            x.with_columns(
                pl.col("value")
                .rolling_max(window_size=w.w, min_samples=1)
                .alias("rolling_max")
            )
            .with_columns((pl.col("value") / pl.col("rolling_max") - 1).alias("ratio"))
            .with_columns(
                pl.col("ratio")
                .rolling_min(window_size=w.w, min_samples=1)
                .alias("value")
            )
            .select(["date", "value"])
        )
    else:
        td = _to_timedelta(w.w)
        vals = x["value"].to_numpy(allow_copy=True)
        dates = x["date"].to_list()
        n = len(vals)
        ratios = np.full(n, np.nan)
        for i, d in enumerate(dates):
            cutoff = d - td
            window_vals = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 0:
                ratios[i] = vals[i] / np.nanmax(window_vals) - 1
        result_arr = np.full(n, np.nan)
        for i, d in enumerate(dates):
            cutoff = d - td
            window_ratios = np.array(
                [
                    ratios[j]
                    for j in range(i + 1)
                    if dates[j] > cutoff and not np.isnan(ratios[j])
                ]
            )
            if len(window_ratios) > 0:
                result_arr[i] = np.nanmin(window_ratios)
        result = pl.DataFrame({"date": dates, "value": result_arr}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )

    return apply_ramp(result, w)


# ---------------------------------------------------------------------------
# CAPM Extensions
# ---------------------------------------------------------------------------


def alpha(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    rf: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Jensen's alpha: intercept of regression of excess returns on benchmark excess returns.

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param rf: annualized risk-free rate
    :param w: rolling window
    :return: rolling alpha (annualized)
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(joined, w)

    x_vals = joined["value"].to_numpy()
    b_vals = joined["value_b"].to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    rf_per_period = rf / ann_factor

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_vals[start : i + 1] - rf_per_period
            bi = b_vals[start : i + 1] - rf_per_period
            mask = ~(np.isnan(xi) | np.isnan(bi))
            xi, bi = xi[mask], bi[mask]
            if len(xi) >= 3 and np.var(bi, ddof=1) > 0:
                b_coef = np.cov(xi, bi, ddof=1)[0, 1] / np.var(bi, ddof=1)
                result[i] = (np.mean(xi) - b_coef * np.mean(bi)) * ann_factor
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx] - rf_per_period
            bi = b_vals[idx] - rf_per_period
            mask = ~(np.isnan(xi) | np.isnan(bi))
            xi, bi = xi[mask], bi[mask]
            if len(xi) >= 3 and np.var(bi, ddof=1) > 0:
                b_coef = np.cov(xi, bi, ddof=1)[0, 1] / np.var(bi, ddof=1)
                result[i] = (np.mean(xi) - b_coef * np.mean(bi)) * ann_factor

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def bull_beta(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling beta in up-market periods (benchmark return > 0).

    :param x: price timeseries
    :param benchmark: benchmark price timeseries
    :param w: rolling window
    :return: rolling bull beta series
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    w = normalize_window(joined, w)

    x_vals = joined["value"].to_numpy()
    b_vals = joined["value_b"].to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_vals[start : i + 1]
            bi = b_vals[start : i + 1]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi > 0)
            xi, bi = xi[mask], bi[mask]
            if len(xi) >= 2 and np.var(bi, ddof=1) > 0:
                result[i] = np.cov(xi, bi, ddof=1)[0, 1] / np.var(bi, ddof=1)
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx]
            bi = b_vals[idx]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi > 0)
            xi, bi = xi[mask], bi[mask]
            if len(xi) >= 2 and np.var(bi, ddof=1) > 0:
                result[i] = np.cov(xi, bi, ddof=1)[0, 1] / np.var(bi, ddof=1)

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def bear_beta(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling beta in down-market periods (benchmark return < 0).

    :param x: price timeseries
    :param benchmark: benchmark price timeseries
    :param w: rolling window
    :return: rolling bear beta series
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    w = normalize_window(joined, w)

    x_vals = joined["value"].to_numpy()
    b_vals = joined["value_b"].to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_vals[start : i + 1]
            bi = b_vals[start : i + 1]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi < 0)
            xi, bi = xi[mask], bi[mask]
            if len(xi) >= 2 and np.var(bi, ddof=1) > 0:
                result[i] = np.cov(xi, bi, ddof=1)[0, 1] / np.var(bi, ddof=1)
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx]
            bi = b_vals[idx]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi < 0)
            xi, bi = xi[mask], bi[mask]
            if len(xi) >= 2 and np.var(bi, ddof=1) > 0:
                result[i] = np.cov(xi, bi, ddof=1)[0, 1] / np.var(bi, ddof=1)

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def timing_ratio(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling timing ratio: bull beta / bear beta. Values > 1 indicate good market timing.

    :param x: price timeseries
    :param benchmark: benchmark price timeseries
    :param w: rolling window
    :return: rolling timing ratio series
    """
    bb = bull_beta(x, benchmark, w)
    be = bear_beta(x, benchmark, w)
    joined = bb.join(be, on="date", how="inner", suffix="_bear")
    return joined.with_columns(
        (pl.col("value") / pl.col("value_bear")).alias("value")
    ).select(["date", "value"])


def tracking_error(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling tracking error: annualized std dev of active returns (portfolio - benchmark).

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param w: rolling window
    :return: tracking error series (in percent, same scale as volatility())
    """
    import math as _math

    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(joined, w)

    active = (joined["value"] - joined["value_b"]).to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            section = active[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                result[i] = (
                    float(np.std(valid, ddof=1)) * _math.sqrt(ann_factor) * 100.0
                )
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([active[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                result[i] = (
                    float(np.std(valid, ddof=1)) * _math.sqrt(ann_factor) * 100.0
                )

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def active_premium(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling active premium: annualized mean active return (portfolio - benchmark).

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param w: rolling window
    :return: annualized active premium series
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(joined, w)

    active = (joined["value"] - joined["value_b"]).to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            section = active[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) >= 1:
                result[i] = float(np.mean(valid)) * ann_factor
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([active[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) >= 1:
                result[i] = float(np.mean(valid)) * ann_factor

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def up_capture(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling up-market capture ratio: portfolio return / benchmark return in up periods.

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param w: rolling window
    :return: up-capture ratio series (1.0 = matching benchmark in up markets)
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    w = normalize_window(joined, w)

    x_vals = joined["value"].to_numpy()
    b_vals = joined["value_b"].to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_vals[start : i + 1]
            bi = b_vals[start : i + 1]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi > 0)
            xi, bi = xi[mask], bi[mask]
            if len(bi) >= 1 and np.mean(bi) != 0:
                result[i] = float(np.mean(xi)) / float(np.mean(bi))
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx]
            bi = b_vals[idx]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi > 0)
            xi, bi = xi[mask], bi[mask]
            if len(bi) >= 1 and np.mean(bi) != 0:
                result[i] = float(np.mean(xi)) / float(np.mean(bi))

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def down_capture(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling down-market capture ratio: portfolio return / benchmark return in down periods.

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param w: rolling window
    :return: down-capture ratio series (< 1.0 is favorable)
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    w = normalize_window(joined, w)

    x_vals = joined["value"].to_numpy()
    b_vals = joined["value_b"].to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_vals[start : i + 1]
            bi = b_vals[start : i + 1]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi < 0)
            xi, bi = xi[mask], bi[mask]
            if len(bi) >= 1 and np.mean(bi) != 0:
                result[i] = float(np.mean(xi)) / float(np.mean(bi))
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx]
            bi = b_vals[idx]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi < 0)
            xi, bi = xi[mask], bi[mask]
            if len(bi) >= 1 and np.mean(bi) != 0:
                result[i] = float(np.mean(xi)) / float(np.mean(bi))

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def up_capture_number(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling up-market capture number: fraction of up benchmark periods where portfolio also rose.

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param w: rolling window
    :return: fraction in [0, 1] series
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    w = normalize_window(joined, w)

    x_vals = joined["value"].to_numpy()
    b_vals = joined["value_b"].to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_vals[start : i + 1]
            bi = b_vals[start : i + 1]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi > 0)
            xi_up = xi[mask]
            if len(xi_up) >= 1:
                result[i] = float(np.mean(xi_up > 0))
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx]
            bi = b_vals[idx]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi > 0)
            xi_up = xi[mask]
            if len(xi_up) >= 1:
                result[i] = float(np.mean(xi_up > 0))

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def down_capture_number(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling down-market capture number: fraction of down benchmark periods where portfolio also fell.

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param w: rolling window
    :return: fraction in [0, 1] series
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    w = normalize_window(joined, w)

    x_vals = joined["value"].to_numpy()
    b_vals = joined["value_b"].to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_vals[start : i + 1]
            bi = b_vals[start : i + 1]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi < 0)
            xi_dn = xi[mask]
            if len(xi_dn) >= 1:
                result[i] = float(np.mean(xi_dn < 0))
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx]
            bi = b_vals[idx]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi < 0)
            xi_dn = xi[mask]
            if len(xi_dn) >= 1:
                result[i] = float(np.mean(xi_dn < 0))

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def up_capture_percent(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling up-market batting average: fraction of up benchmark periods where portfolio outperformed.

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param w: rolling window
    :return: fraction in [0, 1] series
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    w = normalize_window(joined, w)

    x_vals = joined["value"].to_numpy()
    b_vals = joined["value_b"].to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_vals[start : i + 1]
            bi = b_vals[start : i + 1]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi > 0)
            xi_up, bi_up = xi[mask], bi[mask]
            if len(xi_up) >= 1:
                result[i] = float(np.mean(xi_up > bi_up))
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx]
            bi = b_vals[idx]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi > 0)
            xi_up, bi_up = xi[mask], bi[mask]
            if len(xi_up) >= 1:
                result[i] = float(np.mean(xi_up > bi_up))

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def down_capture_percent(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling down-market batting average: fraction of down benchmark periods where portfolio outperformed.

    Outperforming in a down period means losing less than the benchmark (ra > rb).

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param w: rolling window
    :return: fraction in [0, 1] series
    """
    ret_x = returns(x)
    ret_b = returns(benchmark)
    joined = ret_x.join(ret_b, on="date", how="inner", suffix="_b").sort("date")
    w = normalize_window(joined, w)

    x_vals = joined["value"].to_numpy()
    b_vals = joined["value_b"].to_numpy()
    dates = joined["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_vals[start : i + 1]
            bi = b_vals[start : i + 1]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi < 0)
            xi_dn, bi_dn = xi[mask], bi[mask]
            if len(xi_dn) >= 1:
                result[i] = float(np.mean(xi_dn > bi_dn))
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx]
            bi = b_vals[idx]
            mask = (~(np.isnan(xi) | np.isnan(bi))) & (bi < 0)
            xi_dn, bi_dn = xi[mask], bi[mask]
            if len(xi_dn) >= 1:
                result[i] = float(np.mean(xi_dn > bi_dn))

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def persistence_score(
    x: pl.DataFrame,
    period: int,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling persistence score: fraction of sub-periods with positive cumulative returns.

    For each outer rolling window, rolls a sub-window of size `period` and reports the
    fraction of sub-windows where the cumulative return is positive.

    :param x: price timeseries
    :param period: sub-window size in observations
    :param w: outer rolling window
    :return: persistence score in [0, 1] series
    """
    ret = returns(x)
    w = normalize_window(ret, w)

    r_vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    def _score(window_r: np.ndarray) -> Optional[float]:
        m = len(window_r)
        if m < period:
            return None
        pos = 0
        total = 0
        for j in range(m - period + 1):
            sub = window_r[j : j + period]
            if np.any(np.isnan(sub)):
                continue
            total += 1
            if float(np.prod(1.0 + sub)) > 1.0:
                pos += 1
        return pos / total if total > 0 else None

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            s = _score(r_vals[start : i + 1])
            if s is not None:
                result[i] = s
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            s = _score(r_vals[idx])
            if s is not None:
                result[i] = s

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def systematic_risk(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling systematic risk: beta^2 * variance(benchmark).

    :param x: price timeseries
    :param benchmark: benchmark price timeseries
    :param w: rolling window
    :return: systematic risk series
    """
    beta_series = beta(x, benchmark, w, prices=True)
    ret_b = returns(benchmark)
    w_norm = normalize_window(ret_b, w)

    b_vals = ret_b["value"].to_numpy()
    b_dates = ret_b["date"].to_list()
    nb = len(b_dates)
    var_result = np.full(nb, np.nan)

    if isinstance(w_norm.w, int):
        wsize = w_norm.w
        for i in range(nb):
            start = max(0, i - wsize + 1)
            section = b_vals[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                var_result[i] = float(np.var(valid, ddof=1))
    else:
        td = _to_timedelta(w_norm.w)
        for i, d in enumerate(b_dates):
            cutoff = d - td
            section = np.array([b_vals[j] for j in range(i + 1) if b_dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                var_result[i] = float(np.var(valid, ddof=1))

    var_df = pl.DataFrame({"date": b_dates, "value": var_result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    joined = beta_series.join(var_df, on="date", how="inner", suffix="_var")
    return joined.with_columns(
        (pl.col("value") ** 2 * pl.col("value_var")).alias("value")
    ).select(["date", "value"])


def specific_risk(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling specific (idiosyncratic) risk: total variance - systematic variance.

    :param x: price timeseries
    :param benchmark: benchmark price timeseries
    :param w: rolling window
    :return: specific risk series
    """
    from qtk.ts.statistics import var as _var

    ret_x = returns(x)
    w_norm = normalize_window(ret_x, w)
    total_var = _var(ret_x, w_norm)
    sys_risk = systematic_risk(x, benchmark, w_norm)

    joined = total_var.join(sys_risk, on="date", how="inner", suffix="_sys")
    return joined.with_columns(
        (pl.col("value") - pl.col("value_sys")).alias("value")
    ).select(["date", "value"])
