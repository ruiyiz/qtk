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
#   - Removed Goldman Sachs Marquee API dependencies
#   - Ported from pandas.Series to polars.Series / pl.DataFrame
#   - Patched NumPy <2.0 deprecated APIs for NumPy >=2.0 compatibility
#   - Removed Marquee-specific decorators and utilities
#   All mathematical logic is unchanged from the original.
# -----------------------------------------------------------------------

import datetime as dt
import re
from enum import Enum, IntEnum
from typing import Union

import polars as pl

from qtk.errors import QtkValueError

__all__ = [
    "Interpolate",
    "Returns",
    "SeriesType",
    "CurveType",
    "Window",
    "normalize_window",
    "apply_ramp",
]


class Interpolate(Enum):
    INTERSECT = "intersect"
    STEP = "step"
    NAN = "nan"
    ZERO = "zero"
    TIME = "time"


class Returns(Enum):
    SIMPLE = "simple"
    LOGARITHMIC = "logarithmic"
    ABSOLUTE = "absolute"


class SeriesType(Enum):
    PRICES = "prices"
    RETURNS = "returns"


class CurveType(Enum):
    PRICES = "prices"
    EXCESS_RETURNS = "excess_returns"


def _create_enum(name: str, members: list) -> type:
    return Enum(name, {n.upper(): n.lower() for n in members}, module=__name__)


class Window:
    """
    Create a Window with size and ramp up to use.

    :param w: window size
    :param r: ramp up value. Defaults to the window size.
    :return: new window object

    **Usage**

    The window size and ramp up value can either the number of observations or a string representation of the time
    period.

    **Examples**

    Window size is :math:`22` observations and the ramp up value is :math:`10`:

    >>> Window(22, 10)

    Window size is one month and the ramp up size is one week:

    >>> Window('1m', '1w')

    """

    def __init__(self, w: Union[int, str, None] = None, r: Union[int, str, None] = None):
        self.w = w
        self.r = w if r is None else r

    def as_dict(self):
        return {"w": self.w, "r": self.r}

    @classmethod
    def from_dict(cls, obj):
        return Window(w=obj.get("w"), r=obj.get("r"))


def _to_polars_duration(tenor: str) -> str:
    """Convert a gs-quant tenor string to a polars duration string.

    gs-quant: '1h', '5d', '2w', '3m', '1y'
    polars:   '1h', '5d', '2w', '3mo', '1y'

    Note: polars uses 'mo' for months and 'm' for minutes.
    """
    matcher = re.fullmatch(r"(\d+)([hdwmy])", tenor)
    if not matcher:
        raise QtkValueError("invalid tenor " + tenor)
    num = matcher.group(1)
    unit = matcher.group(2)
    unit_map = {"h": "h", "d": "d", "w": "w", "m": "mo", "y": "y"}
    return num + unit_map[unit]


def _to_timedelta(tenor: str) -> dt.timedelta:
    """Convert a tenor string to a Python timedelta.

    For months and years, approximate: 1m ≈ 30 days, 1y ≈ 365 days.
    Used for ramp date arithmetic only.
    """
    matcher = re.fullmatch(r"(\d+)([hdwmy])", tenor)
    if not matcher:
        raise QtkValueError("invalid tenor " + tenor)
    n = int(matcher.group(1))
    unit = matcher.group(2)
    if unit == "h":
        return dt.timedelta(hours=n)
    elif unit == "d":
        return dt.timedelta(days=n)
    elif unit == "w":
        return dt.timedelta(weeks=n)
    elif unit == "m":
        return dt.timedelta(days=n * 30)
    else:
        return dt.timedelta(days=n * 365)


def _tenor_to_month(relative_date: str) -> int:
    matcher = re.fullmatch(r"([1-9]\d*)([my])", relative_date)
    if matcher:
        mag = int(matcher.group(1))
        return mag if matcher.group(2) == "m" else mag * 12
    raise QtkValueError("invalid input: relative date must be in months or years")


def _month_to_tenor(months: int) -> str:
    return f"{months // 12}y" if months % 12 == 0 else f"{months}m"


def _check_window(series_length: int, window: "Window"):
    if series_length > 0 and isinstance(window.w, int) and isinstance(window.r, int):
        if window.w <= 0:
            raise QtkValueError("Window value must be greater than zero.")
        if window.r > series_length or window.r < 0:
            raise QtkValueError(
                "Ramp value must be less than the length of the series and greater than zero."
            )


def normalize_window(
    x: pl.DataFrame,
    window: Union["Window", int, str, None],
    default_window: int = None,
) -> "Window":
    if default_window is None:
        default_window = x.height if isinstance(x, pl.DataFrame) else len(x)

    if isinstance(window, int):
        window = Window(window, window)
    elif isinstance(window, str):
        dur = _to_polars_duration(window)
        window = Window(dur, dur)
    else:
        if window is None:
            window = Window(default_window, 0)
        else:
            w = window.w
            r = window.r
            if isinstance(w, str):
                w = _to_polars_duration(w)
            if isinstance(r, str):
                r = _to_polars_duration(r)
            if w is None:
                w = default_window
            window = Window(w, r)

    _check_window(default_window, window)
    return window


def apply_ramp(x: pl.DataFrame, window: "Window") -> pl.DataFrame:
    _check_window(x.height, window)
    if isinstance(window.w, int) and window.w > x.height:
        return pl.DataFrame({"date": [], "value": []}).cast({"date": pl.Date, "value": pl.Float64})
    if isinstance(window.r, str):
        if x.height == 0:
            return x
        first_date = x["date"][0]
        ramp_td = _to_timedelta(window.r)
        cutoff = first_date + ramp_td
        return x.filter(pl.col("date") >= cutoff)
    else:
        r = window.r if window.r is not None else 0
        return x.slice(r)
