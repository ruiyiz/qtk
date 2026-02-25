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
#   - Removed SIRModel and SEIRModel (depend on lmfit epidemiology module)
#   - Removed Marquee-specific decorators and utilities
#   All mathematical logic is unchanged from the original.
# -----------------------------------------------------------------------

"""
Stats library contains basic statistical operations on timeseries including
rolling aggregations, z-scores, regression, and distribution analysis.
"""

import datetime as dt
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats.mstats as scipy_stats
import statsmodels.api as sm
from scipy.stats import percentileofscore
from statsmodels.regression.rolling import RollingOLS

import polars as pl

from qtk.errors import QtkTypeError, QtkValueError
from qtk.ts.dateops import interpolate
from qtk.ts.helper import (
    Window,
    Interpolate,
    _to_timedelta,
    apply_ramp,
    normalize_window,
)

__all__ = [
    "MeanType",
    "Direction",
    "IntradayDirection",
    "min_",
    "max_",
    "range_",
    "mean",
    "median",
    "mode",
    "sum_",
    "product",
    "std",
    "exponential_std",
    "var",
    "cov",
    "zscores",
    "winsorize",
    "generate_series",
    "generate_series_intraday",
    "percentiles",
    "percentile",
    "LinearRegression",
    "RollingLinearRegression",
]


class MeanType(Enum):
    ARITHMETIC = "arithmetic"
    QUADRATIC = "quadratic"


class Direction(Enum):
    START_TODAY = "start_today"
    END_TODAY = "end_today"


class IntradayDirection(Enum):
    START_INTRADAY_NOW = "start_intraday_now"
    END_INTRADAY_NOW = "end_intraday_now"


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------


def _rolling_apply_np(x: pl.DataFrame, w: Window, fn) -> pl.DataFrame:
    """Apply a numpy function over a rolling integer window."""
    vals = x["value"].to_numpy(allow_copy=True)
    dates = x["date"].to_list()
    wsize = w.w
    result = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        start = max(0, i - wsize + 1)
        section = vals[start : i + 1]
        valid = section[~np.isnan(section)]
        if len(valid) > 0:
            result[i] = fn(valid)
    return pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def _rolling_apply_dur(x: pl.DataFrame, w: Window, fn) -> pl.DataFrame:
    """Apply a numpy function over a rolling duration window."""
    vals = x["value"].to_numpy(allow_copy=True)
    dates = x["date"].to_list()
    td = _to_timedelta(w.w)
    result = np.full(len(vals), np.nan)
    for i, d in enumerate(dates):
        cutoff = d - td
        section = np.array(
            [vals[j] for j in range(i + 1) if dates[j] > cutoff]
        )
        valid = section[~np.isnan(section)]
        if len(valid) > 0:
            result[i] = fn(valid)
    return pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def _rolling_dispatch(x: pl.DataFrame, w: Window, fn_polars_expr, fn_numpy) -> pl.DataFrame:
    """
    Dispatch rolling computation: use polars built-ins for int windows,
    numpy fallback for duration windows.
    """
    if isinstance(w.w, int):
        return x.with_columns(fn_polars_expr(w.w).alias("value"))
    else:
        return _rolling_apply_dur(x, w, fn_numpy)


# ---------------------------------------------------------------------------
# Public rolling aggregation functions
# ---------------------------------------------------------------------------


def min_(
    x: Union[pl.DataFrame, List[pl.DataFrame]],
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Minimum value of series over rolling window.

    :param x: timeseries or list of timeseries
    :param w: window size (observations or tenor string)
    :return: rolling minimum
    """
    if isinstance(x, list):
        x = _merge_list(x, "min")
    w = normalize_window(x, w)
    if isinstance(w.w, int):
        result = x.with_columns(
            pl.col("value").rolling_min(window_size=w.w, min_samples=1).alias("value")
        )
    else:
        result = _rolling_apply_dur(x, w, np.nanmin)
    return apply_ramp(result, w)


def max_(
    x: Union[pl.DataFrame, List[pl.DataFrame]],
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Maximum value of series over rolling window.

    :param x: timeseries or list of timeseries
    :param w: window size
    :return: rolling maximum
    """
    if isinstance(x, list):
        x = _merge_list(x, "max")
    w = normalize_window(x, w)
    if isinstance(w.w, int):
        result = x.with_columns(
            pl.col("value").rolling_max(window_size=w.w, min_samples=1).alias("value")
        )
    else:
        result = _rolling_apply_dur(x, w, np.nanmax)
    return apply_ramp(result, w)


def range_(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Rolling range (max - min) of series.

    :param x: timeseries
    :param w: window size
    :return: rolling range
    """
    w = normalize_window(x, w)
    max_v = max_(x, Window(w.w, 0))
    min_v = min_(x, Window(w.w, 0))
    result = max_v.with_columns(
        (pl.col("value") - min_v["value"]).alias("value")
    )
    return apply_ramp(result, w)


def mean(
    x: Union[pl.DataFrame, List[pl.DataFrame]],
    w: Union[Window, int, str] = Window(None, 0),
    mean_type: MeanType = MeanType.ARITHMETIC,
) -> pl.DataFrame:
    """
    Rolling mean of series.

    :param x: timeseries or list of timeseries
    :param w: window size
    :param mean_type: arithmetic or quadratic mean
    :return: rolling mean
    """
    if isinstance(x, list):
        x = _merge_list(x, "mean")
    w = normalize_window(x, w)

    if mean_type is MeanType.QUADRATIC:
        x = x.with_columns((pl.col("value") ** 2).alias("value"))

    if isinstance(w.w, int):
        result = x.with_columns(
            pl.col("value").rolling_mean(window_size=w.w, min_samples=1).alias("value")
        )
    else:
        result = _rolling_apply_dur(x, w, np.nanmean)

    if mean_type is MeanType.QUADRATIC:
        result = result.with_columns(pl.col("value").sqrt().alias("value"))

    return apply_ramp(result, w)


def median(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Rolling median of series.

    :param x: timeseries
    :param w: window size
    :return: rolling median
    """
    w = normalize_window(x, w)
    if isinstance(w.w, int):
        result = x.with_columns(
            pl.col("value").rolling_median(window_size=w.w, min_samples=1).alias("value")
        )
    else:
        result = _rolling_apply_dur(x, w, np.nanmedian)
    return apply_ramp(result, w)


def mode(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Rolling mode (most common value) of series.

    :param x: timeseries
    :param w: window size
    :return: rolling mode
    """
    w = normalize_window(x, w)

    def _mode(arr):
        valid = arr[~np.isnan(arr)]
        return scipy_stats.mode(valid).mode if len(valid) > 0 else np.nan

    if isinstance(w.w, int):
        result = _rolling_apply_np(x, w, _mode)
    else:
        result = _rolling_apply_dur(x, w, _mode)
    return apply_ramp(result, w)


def sum_(
    x: Union[pl.DataFrame, List[pl.DataFrame]],
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling sum of series.

    :param x: timeseries or list of timeseries
    :param w: window size
    :return: rolling sum
    """
    if isinstance(x, list):
        x = _merge_list(x, "sum")
    w = normalize_window(x, w)
    if isinstance(w.w, int):
        result = x.with_columns(
            pl.col("value").rolling_sum(window_size=w.w, min_samples=1).alias("value")
        )
    else:
        result = _rolling_apply_dur(x, w, np.nansum)
    return apply_ramp(result, w)


def product(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Rolling product of series.

    :param x: timeseries
    :param w: window size
    :return: rolling product
    """
    w = normalize_window(x, w)

    def _prod(arr):
        return np.nanprod(arr)

    if isinstance(w.w, int):
        result = _rolling_apply_np(x, w, _prod)
    else:
        result = _rolling_apply_dur(x, w, _prod)
    return apply_ramp(result, w)


def std(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Rolling standard deviation (unbiased, ddof=1) of series.

    :param x: timeseries
    :param w: window size
    :return: rolling standard deviation
    """
    if x.is_empty():
        return x
    w = normalize_window(x, w)
    if isinstance(w.w, int):
        result = x.with_columns(
            pl.col("value").rolling_std(window_size=w.w, min_samples=1).alias("value")
        )
    else:
        result = _rolling_apply_dur(x, w, lambda a: np.std(a, ddof=1))
    return apply_ramp(result, w)


def exponential_std(x: pl.DataFrame, beta: float = 0.75) -> pl.DataFrame:
    """
    Exponentially weighted standard deviation.

    :param x: timeseries
    :param beta: decay factor (weight on previous value), in [0, 1)
    :return: EWM standard deviation series
    """
    return x.with_columns(
        pl.col("value").ewm_std(alpha=1 - beta, adjust=False).alias("value")
    )


def var(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Rolling variance (unbiased, ddof=1) of series.

    :param x: timeseries
    :param w: window size
    :return: rolling variance
    """
    w = normalize_window(x, w)
    if isinstance(w.w, int):
        result = x.with_columns(
            pl.col("value").rolling_var(window_size=w.w, min_samples=1).alias("value")
        )
    else:
        result = _rolling_apply_dur(x, w, lambda a: np.var(a, ddof=1))
    return apply_ramp(result, w)


def cov(
    x: pl.DataFrame,
    y: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling sample covariance (ddof=1) between two series.

    :param x: first timeseries
    :param y: second timeseries
    :param w: window size
    :return: rolling covariance series
    """
    joined = (
        x.join(y, on="date", how="inner", suffix="_y")
        .sort("date")
    )
    w = normalize_window(joined, w)

    x_arr = joined["value"].to_numpy(allow_copy=True)
    y_arr = joined["value_y"].to_numpy(allow_copy=True)
    dates = joined["date"].to_list()
    n = len(x_arr)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            xi = x_arr[start : i + 1]
            yi = y_arr[start : i + 1]
            mask = ~(np.isnan(xi) | np.isnan(yi))
            xi, yi = xi[mask], yi[mask]
            if len(xi) >= 2:
                result[i] = np.cov(xi, yi, ddof=1)[0, 1]
    else:
        td = _to_timedelta(w.w)
        for i in range(n):
            cutoff = dates[i] - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_arr[idx]
            yi = y_arr[idx]
            mask = ~(np.isnan(xi) | np.isnan(yi))
            xi, yi = xi[mask], yi[mask]
            if len(xi) >= 2:
                result[i] = np.cov(xi, yi, ddof=1)[0, 1]

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def zscores(
    x: pl.DataFrame, w: Union[Window, int, str] = Window(None, 0)
) -> pl.DataFrame:
    """
    Rolling z-scores over a given window.

    :param x: timeseries
    :param w: window size
    :return: rolling z-score series
    """
    if x.height < 1:
        return x

    w = normalize_window(x, w)

    if not w.w:
        if x.height == 1:
            return x.with_columns(pl.lit(0.0).alias("value"))
        clean = x.drop_nulls("value")
        values = clean["value"].to_numpy(allow_copy=True)
        z = scipy_stats.zscore(values, ddof=1)
        z_df = pl.DataFrame({"date": clean["date"].to_list(), "value": z.astype(float)}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )
        return interpolate(z_df, x, Interpolate.NAN)

    if isinstance(w.w, int):
        result = x.with_columns(
            (
                (pl.col("value") - pl.col("value").rolling_mean(window_size=w.w, min_samples=1))
                / pl.col("value").rolling_std(window_size=w.w, min_samples=1)
            ).alias("value")
        )
    else:
        td = _to_timedelta(w.w)
        vals = x["value"].to_numpy(allow_copy=True)
        dates = x["date"].to_list()
        result_arr = np.full(len(vals), np.nan)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                result_arr[i] = scipy_stats.zscore(valid, ddof=1)[-1]
            elif len(valid) == 1:
                result_arr[i] = 0.0
        result = pl.DataFrame({"date": dates, "value": result_arr}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )

    return apply_ramp(result, w)


def winsorize(
    x: pl.DataFrame,
    limit: float = 2.5,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Limit extreme values by capping at mean +/- limit * sigma.

    :param x: timeseries
    :param limit: z-score threshold
    :param w: window (used only for ramp)
    :return: winsorized series
    """
    w = normalize_window(x, w)

    if x.height < 1:
        return x

    mu = x["value"].mean()
    sigma = x["value"].std(ddof=1)
    high = mu + sigma * limit
    low = mu - sigma * limit

    result = x.with_columns(pl.col("value").clip(lower_bound=low, upper_bound=high))
    return apply_ramp(result, w)


def generate_series(
    length: int, direction: Direction = Direction.START_TODAY
) -> pl.DataFrame:
    """
    Generate a random-walk price series.

    :param length: number of observations
    :param direction: START_TODAY or END_TODAY
    :return: date-based timeseries of randomly generated prices
    """
    levels = [100.0]
    first = dt.date.today()
    if direction == Direction.END_TODAY:
        first = first - dt.timedelta(days=length - 1)
    dates = [first]

    rng = np.random.default_rng()
    for i in range(length - 1):
        levels.append(levels[i] + rng.standard_normal())
        dates.append(dt.date.fromordinal(dates[i].toordinal() + 1))

    return pl.DataFrame({"date": dates, "value": levels}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def generate_series_intraday(
    length: int,
    direction: IntradayDirection = IntradayDirection.START_INTRADAY_NOW,
) -> pl.DataFrame:
    """
    Generate a random-walk intraday (minute-level) price series.

    :param length: number of observations (minutes)
    :param direction: START_INTRADAY_NOW or END_INTRADAY_NOW
    :return: datetime-based timeseries of randomly generated prices
    """
    import datetime as _dt

    levels = [100.0]
    now = _dt.datetime.now().replace(second=0, microsecond=0)
    if direction == IntradayDirection.END_INTRADAY_NOW:
        now = now - _dt.timedelta(minutes=length - 1)
    times = [now]

    rng = np.random.default_rng()
    for i in range(length - 1):
        levels.append(levels[i] + rng.standard_normal())
        times.append(times[i] + _dt.timedelta(minutes=1))

    return pl.DataFrame({"datetime": times, "value": levels}).cast(
        {"datetime": pl.Datetime, "value": pl.Float64}
    )


def percentiles(
    x: pl.DataFrame,
    y: Optional[pl.DataFrame] = None,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling percentile rank of y in the distribution of x.

    :param x: distribution series
    :param y: value series (defaults to x)
    :param w: window size
    :return: percentile rank series
    """
    if x.is_empty():
        return x

    if y is None:
        y = x.clone()
    w = normalize_window(y, w)

    if isinstance(w.r, int) and w.r > y.height:
        raise QtkValueError("Ramp value must be less than the length of the series y.")

    if isinstance(w.w, int) and w.w > x.height:
        return pl.DataFrame({"date": [], "value": []}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )

    x_vals = x["value"].to_numpy(allow_copy=True)
    x_dates = x["date"].to_list()
    y_vals = y["value"].to_numpy(allow_copy=True)
    y_dates = y["date"].to_list()
    result_vals = []
    result_dates = []

    for i, (yd, yv) in enumerate(zip(y_dates, y_vals)):
        if isinstance(w.w, int):
            # Find the corresponding position in x
            try:
                xi = x_dates.index(yd)
            except ValueError:
                continue
            start = max(0, xi - w.w + 1)
            sample = x_vals[start : xi + 1]
        else:
            td = _to_timedelta(w.w)
            cutoff = yd - td
            sample = np.array(
                [x_vals[j] for j in range(len(x_dates)) if cutoff < x_dates[j] <= yd]
            )
        valid = sample[~np.isnan(sample)]
        if len(valid) > 0:
            result_vals.append(percentileofscore(valid, yv, kind="mean"))
            result_dates.append(yd)

    df = pl.DataFrame({"date": result_dates, "value": result_vals}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def percentile(
    x: pl.DataFrame,
    n: float,
    w: Union[Window, int, str] = None,
) -> Union[pl.DataFrame, float]:
    """
    nth percentile of a series, either scalar (no window) or rolling.

    :param x: timeseries
    :param n: percentile in [0, 100]
    :param w: window size (None returns a scalar)
    :return: nth percentile value(s)
    """
    if not 0 <= n <= 100:
        raise QtkValueError("percentile must be in range [0, 100]")
    x = x.drop_nulls("value")
    if x.height < 1:
        return x
    if w is None:
        return float(np.percentile(x["value"].to_numpy(allow_copy=True), n))

    w = normalize_window(x, w)
    q = n / 100.0

    if isinstance(w.w, int):
        result = x.with_columns(
            pl.col("value").rolling_quantile(q, window_size=w.w, min_samples=1).alias("value")
        )
    else:
        td = _to_timedelta(w.w)
        vals = x["value"].to_numpy(allow_copy=True)
        dates = x["date"].to_list()
        result_arr = np.full(len(vals), np.nan)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                result_arr[i] = np.quantile(valid, q)
        result = pl.DataFrame({"date": dates, "value": result_arr}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )
    return apply_ramp(result, w)


# ---------------------------------------------------------------------------
# Regression classes
# ---------------------------------------------------------------------------


class LinearRegression:
    """
    Fit an Ordinary Least Squares (OLS) linear regression model.

    :param X: explanatory variable(s) as a DataFrame or list of DataFrames
    :param y: dependent variable DataFrame
    :param fit_intercept: whether to include a constant term
    """

    def __init__(
        self,
        X: Union[pl.DataFrame, List[pl.DataFrame]],
        y: pl.DataFrame,
        fit_intercept: bool = True,
    ):
        if not isinstance(fit_intercept, bool):
            raise QtkTypeError('expected a boolean value for "fit_intercept"')

        X_pd, y_pd = _to_pandas_xy(X, y)

        if fit_intercept:
            X_pd = sm.add_constant(X_pd)
        X_pd.columns = range(len(X_pd.columns))

        X_pd = X_pd[~X_pd.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        y_pd = y_pd[~y_pd.isin([np.nan, np.inf, -np.inf])]
        X_aligned, y_aligned = X_pd.align(y_pd, join="inner", axis=0)

        self._fit_intercept = fit_intercept
        self._res = sm.OLS(y_aligned, X_aligned).fit()

    def coefficient(self, i: int) -> float:
        """Estimated coefficient for predictor i (0 = intercept if used)."""
        return float(self._res.params[i])

    def r_squared(self) -> float:
        """Coefficient of determination."""
        return float(self._res.rsquared)

    def fitted_values(self) -> pl.DataFrame:
        """Fitted values on the training set."""
        fv = self._res.fittedvalues
        return pl.DataFrame({"date": list(fv.index), "value": fv.to_numpy()}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )

    def predict(self, X_predict: Union[pl.DataFrame, List[pl.DataFrame]]) -> pl.DataFrame:
        """Predict using the fitted model."""
        if isinstance(X_predict, list):
            df = pd.concat(
                [pd.Series(d["value"].to_list(), index=d["date"].to_list()) for d in X_predict],
                axis=1,
            )
        else:
            df = pd.Series(
                X_predict["value"].to_list(), index=X_predict["date"].to_list()
            ).to_frame()
        X_pd = sm.add_constant(df) if self._fit_intercept else df
        pred = self._res.predict(X_pd)
        return pl.DataFrame({"date": list(pred.index), "value": pred.to_numpy()}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )

    def standard_deviation_of_errors(self) -> float:
        """Standard deviation of the residuals."""
        return float(np.sqrt(self._res.mse_resid))


class RollingLinearRegression:
    """
    Fit a rolling Ordinary Least Squares (OLS) linear regression model.

    :param X: explanatory variable(s) as a DataFrame or list of DataFrames
    :param y: dependent variable DataFrame
    :param w: rolling window size (number of observations)
    :param fit_intercept: whether to include a constant term
    """

    def __init__(
        self,
        X: Union[pl.DataFrame, List[pl.DataFrame]],
        y: pl.DataFrame,
        w: int,
        fit_intercept: bool = True,
    ):
        if not isinstance(fit_intercept, bool):
            raise QtkTypeError('expected a boolean value for "fit_intercept"')

        X_pd, y_pd = _to_pandas_xy(X, y)

        if fit_intercept:
            X_pd = sm.add_constant(X_pd)
        X_pd.columns = range(len(X_pd.columns))

        if w <= len(X_pd.columns):
            raise QtkValueError("Window length must be larger than the number of explanatory variables")

        X_pd = X_pd[~X_pd.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        y_pd = y_pd[~y_pd.isin([np.nan, np.inf, -np.inf])]
        X_aligned, y_aligned = X_pd.align(y_pd, join="inner", axis=0)

        self._X = X_aligned.copy()
        self._res = RollingOLS(y_aligned, X_aligned, w).fit()

    def coefficient(self, i: int) -> pl.DataFrame:
        """Rolling estimated coefficient for predictor i."""
        params = self._res.params[i]
        return pl.DataFrame({"date": list(params.index), "value": params.to_numpy()}).cast(
            {"date": pl.Date, "value": pl.Float64}
        ).with_columns(pl.col("value").fill_nan(None))

    def r_squared(self) -> pl.DataFrame:
        """Rolling R-squared."""
        rs = self._res.rsquared
        return pl.DataFrame({"date": list(rs.index), "value": rs.to_numpy()}).cast(
            {"date": pl.Date, "value": pl.Float64}
        ).with_columns(pl.col("value").fill_nan(None))

    def fitted_values(self) -> pl.DataFrame:
        """Fitted values at the end of each rolling window."""
        comp = self._X.mul(self._res.params.values)
        fv = comp.sum(axis=1, min_count=len(comp.columns))
        return pl.DataFrame({"date": list(fv.index), "value": fv.to_numpy()}).cast(
            {"date": pl.Date, "value": pl.Float64}
        ).with_columns(pl.col("value").fill_nan(None))

    def standard_deviation_of_errors(self) -> pl.DataFrame:
        """Rolling standard deviation of residuals."""
        se = np.sqrt(self._res.mse_resid)
        return pl.DataFrame({"date": list(se.index), "value": se.to_numpy()}).cast(
            {"date": pl.Date, "value": pl.Float64}
        ).with_columns(pl.col("value").fill_nan(None))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_pandas_xy(
    X: Union[pl.DataFrame, List[pl.DataFrame]], y: pl.DataFrame
) -> tuple:
    """Convert polars DataFrames to aligned pandas Series/DataFrame for statsmodels."""
    if isinstance(X, list):
        X_pd = pd.concat(
            [
                pd.Series(df["value"].to_list(), index=df["date"].to_list(), name=f"x{i}")
                for i, df in enumerate(X)
            ],
            axis=1,
        )
    else:
        X_pd = pd.Series(X["value"].to_list(), index=X["date"].to_list(), name="x0").to_frame()
    y_pd = pd.Series(y["value"].to_list(), index=y["date"].to_list())
    return X_pd, y_pd


def _merge_list(
    series_list: List[pl.DataFrame], agg: str
) -> pl.DataFrame:
    """
    Merge a list of DataFrames with `date` and `value` columns
    by joining on date, then applying a cross-column aggregation.
    """
    if not series_list:
        return pl.DataFrame({"date": [], "value": []}).cast(
            {"date": pl.Date, "value": pl.Float64}
        )
    result = series_list[0].rename({"value": "v0"})
    for i, df in enumerate(series_list[1:], start=1):
        result = result.join(df.rename({"value": f"v{i}"}), on="date", how="outer", coalesce=True)
    result = result.sort("date")
    value_cols = [c for c in result.columns if c != "date"]
    if agg == "min":
        result = result.with_columns(
            pl.min_horizontal(*[pl.col(c) for c in value_cols]).alias("value")
        )
    elif agg == "max":
        result = result.with_columns(
            pl.max_horizontal(*[pl.col(c) for c in value_cols]).alias("value")
        )
    elif agg == "sum":
        result = result.with_columns(
            pl.sum_horizontal(*[pl.col(c) for c in value_cols]).alias("value")
        )
    else:  # mean
        result = result.with_columns(
            pl.mean_horizontal(*[pl.col(c) for c in value_cols]).alias("value")
        )
    return result.select(["date", "value"])
