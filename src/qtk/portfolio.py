"""Portfolio: weighted multi-asset portfolio analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from qtk import panel
from qtk.ts.econometrics import _get_annualization_factor
from qtk.ts.econometrics import beta as _beta
from qtk.ts.econometrics import max_drawdown as _max_drawdown
from qtk.ts.econometrics import returns as _returns
from qtk.ts.econometrics import tracking_error as _tracking_error
from qtk.ts.econometrics import volatility as _volatility
from qtk.ts.ratios import calmar_ratio as _calmar
from qtk.ts.ratios import information_ratio as _info_ratio
from qtk.ts.ratios import sharpe_ratio as _sharpe

__all__ = ["Portfolio"]


def _last_valid(series: pl.DataFrame) -> float | None:
    vals = series["value"].drop_nulls()
    if vals.is_empty():
        return None
    return float(vals[-1])


def _min_valid(series: pl.DataFrame) -> float | None:
    vals = series["value"].drop_nulls()
    if vals.is_empty():
        return None
    return float(vals.min())


@dataclass
class Portfolio:
    weights: dict[str, float]
    benchmark: str | None = None
    name: str = "Portfolio"

    def __post_init__(self):
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")

    def returns(
        self, price_data: pl.DataFrame, field: str = "PX_CLOSE"
    ) -> pl.DataFrame:
        """Weighted portfolio returns as {date, value} DataFrame."""
        tickers = list(self.weights.keys())
        wide = panel.to_wide(price_data, field)

        ret_exprs = [pl.col("date")]
        for t in tickers:
            ret_exprs.append((pl.col(t) / pl.col(t).shift(1) - 1).alias(t))
        ret_wide = wide.select(ret_exprs).slice(1)

        port_expr = pl.lit(0.0)
        for t, w in self.weights.items():
            port_expr = port_expr + pl.col(t) * w

        return ret_wide.select([pl.col("date"), port_expr.alias("value")])

    def cumulative(
        self, price_data: pl.DataFrame, field: str = "PX_CLOSE"
    ) -> pl.DataFrame:
        """Cumulative return series indexed to 1.0."""
        ret = self.returns(price_data, field)
        return ret.with_columns(((pl.col("value") + 1).cum_prod()).alias("value"))

    def contribution(
        self, price_data: pl.DataFrame, field: str = "PX_CLOSE"
    ) -> pl.DataFrame:
        """Per-security contribution to portfolio returns.

        Returns DataFrame: Ticker, date, value
        (contribution = weight * security return).
        """
        tickers = list(self.weights.keys())
        wide = panel.to_wide(price_data, field)

        ret_exprs = [pl.col("date")]
        for t in tickers:
            ret_exprs.append((pl.col(t) / pl.col(t).shift(1) - 1).alias(t))
        ret_wide = wide.select(ret_exprs).slice(1)

        parts = []
        for t, w in self.weights.items():
            part = ret_wide.select(
                [
                    pl.lit(t).alias("Ticker"),
                    pl.col("date"),
                    (pl.col(t) * w).alias("value"),
                ]
            )
            parts.append(part)

        return pl.concat(parts).select(["Ticker", "date", "value"])

    def summary(self, price_data: pl.DataFrame, field: str = "PX_CLOSE") -> dict:
        """Key portfolio metrics as a dict.

        Includes: total_return, annualized_return, volatility, sharpe,
        max_drawdown, calmar. If benchmark is set and present in price_data:
        tracking_error, info_ratio, beta.
        """
        ret = self.returns(price_data, field)
        cum = self.cumulative(price_data, field)
        if cum.is_empty():
            return {}

        total_return = float(cum["value"][-1]) - 1.0

        dates = cum["date"].to_list()
        n_days = (dates[-1] - dates[0]).days
        ann_return = (
            (1.0 + total_return) ** (365.25 / n_days) - 1.0
            if n_days > 0
            else float("nan")
        )

        # Pass returns directly (returns_type=None) to avoid the off-by-one
        # window issue when volatility normalizes window from the price series
        # then applies it to returns (which are one row shorter).
        ann_factor = _get_annualization_factor(ret)
        vol_series = _volatility(
            ret, returns_type=None, annualization_factor=ann_factor
        )
        vol = (_last_valid(vol_series) or float("nan")) / 100.0

        sharpe_series = _sharpe(cum)
        sharpe = _last_valid(sharpe_series) or float("nan")

        dd_series = _max_drawdown(cum)
        max_dd = _min_valid(dd_series) or float("nan")

        calmar_series = _calmar(cum)
        calmar = _last_valid(calmar_series) or float("nan")

        result: dict = {
            "total_return": total_return,
            "annualized_return": ann_return,
            "volatility": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
        }

        if self.benchmark and "Ticker" in price_data.columns:
            bench_tickers = price_data["Ticker"].unique().to_list()
            if self.benchmark in bench_tickers:
                bench_series = panel.to_series(price_data, self.benchmark, field)
                bench_ret = _returns(bench_series)
                bench_cum = bench_ret.with_columns(
                    ((pl.col("value") + 1).cum_prod()).alias("value")
                )
                te_series = _tracking_error(cum, bench_cum)
                result["tracking_error"] = (
                    _last_valid(te_series) or float("nan")
                ) / 100.0

                ir_series = _info_ratio(cum, bench_cum)
                result["info_ratio"] = _last_valid(ir_series) or float("nan")

                beta_series = _beta(cum, bench_cum)
                result["beta"] = _last_valid(beta_series) or float("nan")

        return result
