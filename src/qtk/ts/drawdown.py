"""
Drawdown analysis: full drawdown series, event identification, and statistics.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import polars as pl

from qtk.errors import QtkValueError

__all__ = [
    "drawdowns",
    "find_drawdowns",
    "sort_drawdowns",
    "average_drawdown",
    "average_drawdown_length",
    "average_recovery",
    "drawdown_deviation",
    "conditional_drawdown",
]


def drawdowns(x: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the drawdown time series: peak-to-current decline at each point.

    :param x: price timeseries
    :return: drawdown series (0 at peaks, negative elsewhere)
    """
    vals = x["value"].to_numpy(allow_copy=True)
    dates = x["date"].to_list()
    n = len(vals)
    dd = np.full(n, np.nan)
    running_max = np.nan
    for i in range(n):
        v = vals[i]
        if np.isnan(v):
            continue
        if np.isnan(running_max) or v > running_max:
            running_max = v
        dd[i] = v / running_max - 1.0
    return pl.DataFrame({"date": dates, "value": dd}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def find_drawdowns(x: pl.DataFrame) -> List[dict]:
    """
    Identify all drawdown events.

    :param x: price timeseries
    :return: list of dicts with keys:
        start (date), trough (date), end (date or None if ongoing),
        depth (float, negative), length (int days), recovery (int days or None)
    """
    dd_series = drawdowns(x)
    dd_vals = dd_series["value"].to_numpy(allow_copy=True)
    dates = x["date"].to_list()
    n = len(dates)

    events = []
    in_dd = False
    start_idx = None
    trough_idx = None
    trough_val = 0.0

    for i in range(n):
        v = dd_vals[i]
        if np.isnan(v):
            continue
        if v < 0 and not in_dd:
            in_dd = True
            start_idx = i - 1 if i > 0 else i
            trough_idx = i
            trough_val = v
        elif in_dd:
            if v < trough_val:
                trough_idx = i
                trough_val = v
            if v == 0.0:
                end_idx = i
                recovery = (dates[end_idx] - dates[trough_idx]).days
                events.append(
                    {
                        "start": dates[start_idx],
                        "trough": dates[trough_idx],
                        "end": dates[end_idx],
                        "depth": trough_val,
                        "length": (dates[trough_idx] - dates[start_idx]).days,
                        "recovery": recovery,
                    }
                )
                in_dd = False
                trough_val = 0.0

    if in_dd:
        events.append(
            {
                "start": dates[start_idx],
                "trough": dates[trough_idx],
                "end": None,
                "depth": trough_val,
                "length": (dates[trough_idx] - dates[start_idx]).days,
                "recovery": None,
            }
        )

    return events


def sort_drawdowns(x: pl.DataFrame, n: Optional[int] = None) -> List[dict]:
    """
    Return drawdown events sorted by depth (worst first).

    :param x: price timeseries
    :param n: number of top drawdowns to return (None = all)
    :return: sorted list of drawdown event dicts
    """
    events = find_drawdowns(x)
    events.sort(key=lambda e: e["depth"])
    return events[:n] if n is not None else events


def average_drawdown(x: pl.DataFrame) -> float:
    """
    Mean depth across all drawdown events.

    :param x: price timeseries
    :return: average drawdown depth (negative float)
    """
    events = find_drawdowns(x)
    if not events:
        return 0.0
    return float(np.mean([e["depth"] for e in events]))


def average_drawdown_length(x: pl.DataFrame) -> float:
    """
    Mean duration (peak to trough) in days across all drawdown events.

    :param x: price timeseries
    :return: average drawdown length in days
    """
    events = find_drawdowns(x)
    if not events:
        return 0.0
    return float(np.mean([e["length"] for e in events]))


def average_recovery(x: pl.DataFrame) -> float:
    """
    Mean recovery time (trough to recovery) in days for completed drawdowns.

    :param x: price timeseries
    :return: average recovery time in days
    """
    events = find_drawdowns(x)
    completed = [e["recovery"] for e in events if e["recovery"] is not None]
    if not completed:
        return 0.0
    return float(np.mean(completed))


def drawdown_deviation(x: pl.DataFrame) -> float:
    """
    Standard deviation of drawdown depths across all drawdown events.

    :param x: price timeseries
    :return: standard deviation of drawdown depths
    """
    events = find_drawdowns(x)
    if len(events) < 2:
        return 0.0
    return float(np.std([e["depth"] for e in events], ddof=1))


def conditional_drawdown(x: pl.DataFrame, p: float = 0.05) -> float:
    """
    Conditional Drawdown at Risk (CDaR): mean of the worst p-fraction of drawdowns.

    :param x: price timeseries
    :param p: tail probability (e.g. 0.05 for worst 5%)
    :return: CDaR (negative float)
    """
    if not 0 < p <= 1:
        raise QtkValueError("p must be in (0, 1]")
    dd_vals = drawdowns(x)["value"].drop_nulls().to_numpy()
    dd_vals = dd_vals[dd_vals < 0]
    if len(dd_vals) == 0:
        return 0.0
    cutoff = np.quantile(dd_vals, p)
    tail = dd_vals[dd_vals <= cutoff]
    return float(np.mean(tail))
