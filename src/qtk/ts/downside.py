"""
Downside risk measures: semi-deviation, LPM, pain/ulcer indices, and related metrics.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import polars as pl

from qtk.errors import QtkValueError
from qtk.ts.helper import Window, apply_ramp, normalize_window, _to_timedelta
from qtk.ts.econometrics import returns as _returns, _get_annualization_factor

__all__ = [
    "downside_deviation",
    "semi_deviation",
    "semi_variance",
    "lpm",
    "upside_potential_ratio",
    "pain_index",
    "ulcer_index",
    "downside_frequency",
    "upside_frequency",
    "volatility_skewness",
]


def _get_returns(x: pl.DataFrame) -> np.ndarray:
    """Convert price series to simple returns array."""
    ret = _returns(x)
    return ret["value"].to_numpy(allow_copy=True)


def downside_deviation(
    x: pl.DataFrame,
    mar: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling downside deviation: annualized std dev of returns below MAR.

    :param x: price timeseries
    :param mar: minimum acceptable return (per-period, e.g. 0 for zero)
    :param w: rolling window
    :return: annualized downside deviation series (same scale as volatility())
    """
    ret = _returns(x)
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(ret, w)

    vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    n = len(vals)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            section = vals[start : i + 1]
            below = section[section < mar]
            if len(below) > 0:
                result[i] = math.sqrt(np.mean((below - mar) ** 2) * ann_factor) * 100.0
            else:
                result[i] = 0.0
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            below = section[section < mar]
            if len(below) > 0:
                result[i] = math.sqrt(np.mean((below - mar) ** 2) * ann_factor) * 100.0
            else:
                result[i] = 0.0

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def semi_deviation(
    x: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling semi-deviation: annualized std dev of returns below their mean.

    :param x: price timeseries
    :param w: rolling window
    :return: annualized semi-deviation series
    """
    ret = _returns(x)
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(ret, w)

    vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    n = len(vals)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            section = vals[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                mu = np.mean(valid)
                below = valid[valid < mu]
                if len(below) > 0:
                    result[i] = (
                        math.sqrt(np.mean((below - mu) ** 2) * ann_factor) * 100.0
                    )
                else:
                    result[i] = 0.0
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                mu = np.mean(valid)
                below = valid[valid < mu]
                if len(below) > 0:
                    result[i] = (
                        math.sqrt(np.mean((below - mu) ** 2) * ann_factor) * 100.0
                    )
                else:
                    result[i] = 0.0

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def semi_variance(
    x: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling semi-variance: annualized variance of returns below their mean.

    :param x: price timeseries
    :param w: rolling window
    :return: annualized semi-variance series
    """
    sd = semi_deviation(x, w)
    return sd.with_columns(((pl.col("value") / 100.0) ** 2).alias("value"))


def lpm(
    x: pl.DataFrame,
    n: int = 2,
    threshold: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Lower Partial Moment of order n below threshold.

    LPM(n, threshold) = mean(max(threshold - R, 0)^n)

    :param x: price timeseries
    :param n: moment order (1=expected shortfall below threshold, 2=downside variance)
    :param threshold: return threshold
    :param w: rolling window
    :return: LPM series
    """
    ret = _returns(x)
    w = normalize_window(ret, w)

    vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    result = np.full(len(vals), np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(len(vals)):
            start = max(0, i - wsize + 1)
            section = vals[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                result[i] = np.mean(np.maximum(threshold - valid, 0) ** n)
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                result[i] = np.mean(np.maximum(threshold - valid, 0) ** n)

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def upside_potential_ratio(
    x: pl.DataFrame,
    mar: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Upside Potential Ratio: UPM(1, mar) / sqrt(LPM(2, mar)).

    :param x: price timeseries
    :param mar: minimum acceptable return
    :param w: rolling window
    :return: upside potential ratio series
    """
    ret = _returns(x)
    w = normalize_window(ret, w)

    vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    result = np.full(len(vals), np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(len(vals)):
            start = max(0, i - wsize + 1)
            section = vals[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                upm1 = np.mean(np.maximum(valid - mar, 0))
                lpm2 = np.mean(np.maximum(mar - valid, 0) ** 2)
                denom = math.sqrt(lpm2)
                result[i] = upm1 / denom if denom > 0 else np.nan
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                upm1 = np.mean(np.maximum(valid - mar, 0))
                lpm2 = np.mean(np.maximum(mar - valid, 0) ** 2)
                denom = math.sqrt(lpm2)
                result[i] = upm1 / denom if denom > 0 else np.nan

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def pain_index(x: pl.DataFrame) -> float:
    """
    Pain Index: mean absolute drawdown depth over the full series.

    :param x: price timeseries
    :return: mean absolute drawdown (positive float)
    """
    from qtk.ts.drawdown import drawdowns as _drawdowns

    dd = _drawdowns(x)["value"].drop_nulls().to_numpy()
    return float(np.mean(np.abs(dd)))


def ulcer_index(x: pl.DataFrame) -> float:
    """
    Ulcer Index: root mean square of drawdown depths.

    :param x: price timeseries
    :return: ulcer index (positive float)
    """
    from qtk.ts.drawdown import drawdowns as _drawdowns

    dd = _drawdowns(x)["value"].drop_nulls().to_numpy()
    return float(math.sqrt(np.mean(dd**2)))


def downside_frequency(
    x: pl.DataFrame,
    mar: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling fraction of return periods below MAR.

    :param x: price timeseries
    :param mar: minimum acceptable return
    :param w: rolling window
    :return: series of fractions in [0, 1]
    """
    ret = _returns(x)
    w = normalize_window(ret, w)

    vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    result = np.full(len(vals), np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(len(vals)):
            start = max(0, i - wsize + 1)
            section = vals[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                result[i] = np.sum(valid < mar) / len(valid)
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                result[i] = np.sum(valid < mar) / len(valid)

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def upside_frequency(
    x: pl.DataFrame,
    mar: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling fraction of return periods above MAR.

    :param x: price timeseries
    :param mar: minimum acceptable return
    :param w: rolling window
    :return: series of fractions in [0, 1]
    """
    ret = _returns(x)
    w = normalize_window(ret, w)

    vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    result = np.full(len(vals), np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(len(vals)):
            start = max(0, i - wsize + 1)
            section = vals[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                result[i] = np.sum(valid > mar) / len(valid)
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                result[i] = np.sum(valid > mar) / len(valid)

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def volatility_skewness(
    x: pl.DataFrame,
    mar: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling volatility skewness: upside vol / downside vol.

    Upside vol = std dev of returns above MAR; downside vol = std dev below MAR.

    :param x: price timeseries
    :param mar: minimum acceptable return
    :param w: rolling window
    :return: volatility skewness series (>1 means more upside vol)
    """
    ret = _returns(x)
    w = normalize_window(ret, w)

    vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    result = np.full(len(vals), np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(len(vals)):
            start = max(0, i - wsize + 1)
            section = vals[start : i + 1]
            valid = section[~np.isnan(section)]
            above = valid[valid > mar]
            below = valid[valid < mar]
            if len(above) >= 2 and len(below) >= 2:
                up_vol = np.std(above, ddof=1)
                down_vol = np.std(below, ddof=1)
                result[i] = up_vol / down_vol if down_vol > 0 else np.nan
    else:
        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            above = valid[valid > mar]
            below = valid[valid < mar]
            if len(above) >= 2 and len(below) >= 2:
                up_vol = np.std(above, ddof=1)
                down_vol = np.std(below, ddof=1)
                result[i] = up_vol / down_vol if down_vol > 0 else np.nan

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)
