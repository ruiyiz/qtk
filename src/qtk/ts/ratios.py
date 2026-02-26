"""
Risk-adjusted performance ratios: Sharpe, Sortino, Calmar, Information, Treynor, Omega,
Sterling, Burke, and Martin ratios.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import polars as pl

from qtk.errors import QtkValueError
from qtk.ts.helper import Window, apply_ramp, normalize_window
from qtk.ts.econometrics import (
    returns as _returns,
    volatility as _volatility,
    beta as _beta,
    max_drawdown as _max_drawdown,
    _get_annualization_factor,
)

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "information_ratio",
    "treynor_ratio",
    "omega_ratio",
    "sterling_ratio",
    "burke_ratio",
    "martin_ratio",
]


def _annualized_mean_return(ret_vals: np.ndarray, ann_factor: int) -> float:
    """Mean per-period return annualized."""
    return float(np.mean(ret_vals)) * ann_factor


def sharpe_ratio(
    x: pl.DataFrame,
    rf: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Sharpe ratio: (annualized return - rf) / annualized volatility.

    :param x: price timeseries
    :param rf: annualized risk-free rate (e.g. 0.04 for 4%)
    :param w: rolling window
    :return: Sharpe ratio series
    """
    ret = _returns(x)
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(ret, w)

    vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    n = len(vals)
    result = np.full(n, np.nan)

    rf_per_period = rf / ann_factor

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            section = vals[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                excess = valid - rf_per_period
                ann_ret = float(np.mean(excess)) * ann_factor
                ann_vol = float(np.std(valid, ddof=1)) * math.sqrt(ann_factor)
                result[i] = ann_ret / ann_vol if ann_vol > 0 else np.nan
    else:
        from qtk.ts.helper import _to_timedelta

        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                excess = valid - rf_per_period
                ann_ret = float(np.mean(excess)) * ann_factor
                ann_vol = float(np.std(valid, ddof=1)) * math.sqrt(ann_factor)
                result[i] = ann_ret / ann_vol if ann_vol > 0 else np.nan

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def sortino_ratio(
    x: pl.DataFrame,
    mar: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Sortino ratio: annualized excess return / annualized downside deviation.

    :param x: price timeseries
    :param mar: minimum acceptable return (annualized)
    :param w: rolling window
    :return: Sortino ratio series
    """
    ret = _returns(x)
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(ret, w)

    vals = ret["value"].to_numpy(allow_copy=True)
    dates = ret["date"].to_list()
    n = len(vals)
    result = np.full(n, np.nan)

    mar_per_period = mar / ann_factor

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            section = vals[start : i + 1]
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                ann_ret = (float(np.mean(valid)) - mar_per_period) * ann_factor
                below = valid[valid < mar_per_period]
                if len(below) > 0:
                    dd = math.sqrt(np.mean((below - mar_per_period) ** 2) * ann_factor)
                    result[i] = ann_ret / dd if dd > 0 else np.nan
                else:
                    result[i] = np.nan
    else:
        from qtk.ts.helper import _to_timedelta

        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                ann_ret = (float(np.mean(valid)) - mar_per_period) * ann_factor
                below = valid[valid < mar_per_period]
                if len(below) > 0:
                    dd = math.sqrt(np.mean((below - mar_per_period) ** 2) * ann_factor)
                    result[i] = ann_ret / dd if dd > 0 else np.nan
                else:
                    result[i] = np.nan

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def calmar_ratio(
    x: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Calmar ratio: annualized return / abs(maximum drawdown).

    :param x: price timeseries
    :param w: rolling window
    :return: Calmar ratio series
    """
    ret = _returns(x)
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(x, w)

    vals_p = x["value"].to_numpy(allow_copy=True)
    ret_vals = ret["value"].to_numpy(allow_copy=True)
    dates = x["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            prices_window = vals_p[start : i + 1]
            ret_window = ret_vals[max(0, start - 1 + 1) : i]
            valid_ret = ret_window[~np.isnan(ret_window)]
            if len(valid_ret) < 1 or len(prices_window) < 2:
                continue
            ann_ret = float(np.mean(valid_ret)) * ann_factor
            running_max = np.maximum.accumulate(prices_window)
            dd = prices_window / running_max - 1.0
            max_dd = float(np.nanmin(dd))
            if max_dd < 0:
                result[i] = ann_ret / abs(max_dd)
    else:
        from qtk.ts.helper import _to_timedelta

        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            if len(idx) < 2:
                continue
            prices_window = vals_p[idx]
            ret_window = np.array(
                [
                    ret_vals[j - 1]
                    for j in idx
                    if j > 0 and not np.isnan(ret_vals[j - 1])
                ]
            )
            if len(ret_window) < 1:
                continue
            ann_ret = float(np.mean(ret_window)) * ann_factor
            running_max = np.maximum.accumulate(prices_window)
            dd = prices_window / running_max - 1.0
            max_dd = float(np.nanmin(dd))
            if max_dd < 0:
                result[i] = ann_ret / abs(max_dd)

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def information_ratio(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Information ratio: annualized active return / tracking error.

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param w: rolling window
    :return: information ratio series
    """
    ret_x = _returns(x)
    ret_b = _returns(benchmark)
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
                ann_active = float(np.mean(valid)) * ann_factor
                tracking_err = float(np.std(valid, ddof=1)) * math.sqrt(ann_factor)
                result[i] = ann_active / tracking_err if tracking_err > 0 else np.nan
    else:
        from qtk.ts.helper import _to_timedelta

        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([active[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) >= 2:
                ann_active = float(np.mean(valid)) * ann_factor
                tracking_err = float(np.std(valid, ddof=1)) * math.sqrt(ann_factor)
                result[i] = ann_active / tracking_err if tracking_err > 0 else np.nan

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def treynor_ratio(
    x: pl.DataFrame,
    benchmark: pl.DataFrame,
    rf: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Treynor ratio: annualized excess return / beta.

    :param x: price timeseries of the portfolio
    :param benchmark: price timeseries of the benchmark
    :param rf: annualized risk-free rate
    :param w: rolling window
    :return: Treynor ratio series
    """
    ret_x = _returns(x)
    ret_b = _returns(benchmark)
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
            xi = x_vals[start : i + 1]
            bi = b_vals[start : i + 1]
            mask = ~(np.isnan(xi) | np.isnan(bi))
            xi, bi = xi[mask], bi[mask]
            if len(xi) >= 2:
                beta_val = (
                    np.cov(xi, bi, ddof=1)[0, 1] / np.var(bi, ddof=1)
                    if np.var(bi, ddof=1) > 0
                    else np.nan
                )
                if beta_val and not np.isnan(beta_val):
                    ann_ret = (float(np.mean(xi)) - rf_per_period) * ann_factor
                    result[i] = ann_ret / beta_val
    else:
        from qtk.ts.helper import _to_timedelta

        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            xi = x_vals[idx]
            bi = b_vals[idx]
            mask = ~(np.isnan(xi) | np.isnan(bi))
            xi, bi = xi[mask], bi[mask]
            if len(xi) >= 2:
                beta_val = (
                    np.cov(xi, bi, ddof=1)[0, 1] / np.var(bi, ddof=1)
                    if np.var(bi, ddof=1) > 0
                    else np.nan
                )
                if beta_val and not np.isnan(beta_val):
                    ann_ret = (float(np.mean(xi)) - rf_per_period) * ann_factor
                    result[i] = ann_ret / beta_val

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def omega_ratio(
    x: pl.DataFrame,
    threshold: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Omega ratio: sum of gains above threshold / sum of losses below threshold.

    :param x: price timeseries
    :param threshold: return threshold (per-period, e.g. 0)
    :param w: rolling window
    :return: Omega ratio series (> 1 is favorable)
    """
    ret = _returns(x)
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
            if len(valid) > 0:
                gains = np.sum(np.maximum(valid - threshold, 0))
                losses = np.sum(np.maximum(threshold - valid, 0))
                result[i] = gains / losses if losses > 0 else np.nan
    else:
        from qtk.ts.helper import _to_timedelta

        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            section = np.array([vals[j] for j in range(i + 1) if dates[j] > cutoff])
            valid = section[~np.isnan(section)]
            if len(valid) > 0:
                gains = np.sum(np.maximum(valid - threshold, 0))
                losses = np.sum(np.maximum(threshold - valid, 0))
                result[i] = gains / losses if losses > 0 else np.nan

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def sterling_ratio(
    x: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Sterling ratio: annualized return / average drawdown depth.

    :param x: price timeseries
    :param w: rolling window
    :return: Sterling ratio series
    """
    from qtk.ts.drawdown import find_drawdowns as _find_drawdowns

    ret = _returns(x)
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(x, w)

    vals_p = x["value"].to_numpy(allow_copy=True)
    ret_vals = ret["value"].to_numpy(allow_copy=True)
    dates = x["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            prices_window = vals_p[start : i + 1]
            ret_window = ret_vals[start:i] if i > start else np.array([])
            valid_ret = (
                ret_window[~np.isnan(ret_window)]
                if len(ret_window) > 0
                else np.array([])
            )
            if len(valid_ret) < 1 or len(prices_window) < 2:
                continue
            ann_ret = float(np.mean(valid_ret)) * ann_factor
            running_max = np.maximum.accumulate(prices_window)
            dd_vals = prices_window / running_max - 1.0
            neg_dds = dd_vals[dd_vals < 0]
            if len(neg_dds) > 0:
                avg_dd = float(np.mean(np.abs(neg_dds)))
                result[i] = ann_ret / avg_dd if avg_dd > 0 else np.nan
    else:
        from qtk.ts.helper import _to_timedelta

        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            if len(idx) < 2:
                continue
            prices_window = vals_p[idx]
            ret_window = np.array(
                [
                    ret_vals[j - 1]
                    for j in idx
                    if j > 0 and not np.isnan(ret_vals[j - 1])
                ]
            )
            if len(ret_window) < 1:
                continue
            ann_ret = float(np.mean(ret_window)) * ann_factor
            running_max = np.maximum.accumulate(prices_window)
            dd_vals = prices_window / running_max - 1.0
            neg_dds = dd_vals[dd_vals < 0]
            if len(neg_dds) > 0:
                avg_dd = float(np.mean(np.abs(neg_dds)))
                result[i] = ann_ret / avg_dd if avg_dd > 0 else np.nan

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def burke_ratio(
    x: pl.DataFrame,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Burke ratio: annualized return / sqrt(sum of squared drawdowns).

    :param x: price timeseries
    :param w: rolling window
    :return: Burke ratio series
    """
    ret = _returns(x)
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(x, w)

    vals_p = x["value"].to_numpy(allow_copy=True)
    ret_vals = ret["value"].to_numpy(allow_copy=True)
    dates = x["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            prices_window = vals_p[start : i + 1]
            ret_window = ret_vals[start:i] if i > start else np.array([])
            valid_ret = (
                ret_window[~np.isnan(ret_window)]
                if len(ret_window) > 0
                else np.array([])
            )
            if len(valid_ret) < 1 or len(prices_window) < 2:
                continue
            ann_ret = float(np.mean(valid_ret)) * ann_factor
            running_max = np.maximum.accumulate(prices_window)
            dd_vals = prices_window / running_max - 1.0
            neg_dds = dd_vals[dd_vals < 0]
            if len(neg_dds) > 0:
                denom = math.sqrt(float(np.sum(neg_dds**2)))
                result[i] = ann_ret / denom if denom > 0 else np.nan
    else:
        from qtk.ts.helper import _to_timedelta

        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            if len(idx) < 2:
                continue
            prices_window = vals_p[idx]
            ret_window = np.array(
                [
                    ret_vals[j - 1]
                    for j in idx
                    if j > 0 and not np.isnan(ret_vals[j - 1])
                ]
            )
            if len(ret_window) < 1:
                continue
            ann_ret = float(np.mean(ret_window)) * ann_factor
            running_max = np.maximum.accumulate(prices_window)
            dd_vals = prices_window / running_max - 1.0
            neg_dds = dd_vals[dd_vals < 0]
            if len(neg_dds) > 0:
                denom = math.sqrt(float(np.sum(neg_dds**2)))
                result[i] = ann_ret / denom if denom > 0 else np.nan

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)


def martin_ratio(
    x: pl.DataFrame,
    rf: float = 0.0,
    w: Union[Window, int, str] = Window(None, 0),
) -> pl.DataFrame:
    """
    Rolling Martin ratio (Ulcer Performance Index): annualized excess return / ulcer index.

    :param x: price timeseries
    :param rf: annualized risk-free rate
    :param w: rolling window
    :return: Martin ratio series
    """
    ret = _returns(x)
    ann_factor = _get_annualization_factor(x)
    w = normalize_window(x, w)

    vals_p = x["value"].to_numpy(allow_copy=True)
    ret_vals = ret["value"].to_numpy(allow_copy=True)
    dates = x["date"].to_list()
    n = len(dates)
    result = np.full(n, np.nan)

    rf_per_period = rf / ann_factor

    if isinstance(w.w, int):
        wsize = w.w
        for i in range(n):
            start = max(0, i - wsize + 1)
            prices_window = vals_p[start : i + 1]
            ret_window = ret_vals[start:i] if i > start else np.array([])
            valid_ret = (
                ret_window[~np.isnan(ret_window)]
                if len(ret_window) > 0
                else np.array([])
            )
            if len(valid_ret) < 1 or len(prices_window) < 2:
                continue
            ann_ret = (float(np.mean(valid_ret)) - rf_per_period) * ann_factor
            running_max = np.maximum.accumulate(prices_window)
            dd_vals = prices_window / running_max - 1.0
            ulcer = math.sqrt(float(np.mean(dd_vals**2)))
            result[i] = ann_ret / ulcer if ulcer > 0 else np.nan
    else:
        from qtk.ts.helper import _to_timedelta

        td = _to_timedelta(w.w)
        for i, d in enumerate(dates):
            cutoff = d - td
            idx = [j for j in range(i + 1) if dates[j] > cutoff]
            if len(idx) < 2:
                continue
            prices_window = vals_p[idx]
            ret_window = np.array(
                [
                    ret_vals[j - 1]
                    for j in idx
                    if j > 0 and not np.isnan(ret_vals[j - 1])
                ]
            )
            if len(ret_window) < 1:
                continue
            ann_ret = (float(np.mean(ret_window)) - rf_per_period) * ann_factor
            running_max = np.maximum.accumulate(prices_window)
            dd_vals = prices_window / running_max - 1.0
            ulcer = math.sqrt(float(np.mean(dd_vals**2)))
            result[i] = ann_ret / ulcer if ulcer > 0 else np.nan

    df = pl.DataFrame({"date": dates, "value": result}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )
    return apply_ramp(df, w)
