"""PositionHistory: time-varying multi-asset portfolio analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from qtk.portfolio import Portfolio

__all__ = ["PositionHistory"]


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
class PositionHistory:
    """Time-varying portfolio analytics from position snapshots.

    Parameters
    ----------
    holdings:
        DataFrame with columns: date (Date), ticker (Str), shares (Float64).
        Dates need not be consecutive; forward-fill is applied internally.
    name:
        Display name for the portfolio.
    benchmark:
        Ticker for benchmark comparisons (must be present in price_data).
    """

    holdings: pl.DataFrame
    name: str = "Portfolio"
    benchmark: str | None = None

    def __post_init__(self):
        required = {"date", "ticker", "shares"}
        missing = required - set(self.holdings.columns)
        if missing:
            raise ValueError(
                f"holdings missing required columns: {', '.join(sorted(missing))}"
            )

    def _fill_holdings(self, price_data: pl.DataFrame) -> pl.DataFrame:
        """Forward-fill sparse holdings across all dates present in price_data."""
        available_tickers = price_data["Ticker"].unique().to_list()
        holdings = self.holdings.filter(pl.col("ticker").is_in(available_tickers))

        if holdings.is_empty():
            return pl.DataFrame(
                schema={"date": pl.Date, "ticker": pl.String, "shares": pl.Float64}
            )

        dates = price_data["ValueDate"].unique().sort().to_frame("date")
        tickers = holdings["ticker"].unique().to_frame("ticker")

        grid = dates.join(tickers, how="cross")
        filled = grid.join(holdings, on=["date", "ticker"], how="left")
        filled = filled.sort(["ticker", "date"]).with_columns(
            pl.col("shares").forward_fill().over("ticker")
        )
        filled = filled.with_columns(pl.col("shares").fill_null(0.0))
        return filled.filter(pl.col("shares") != 0.0)

    def weights(
        self, price_data: pl.DataFrame, field: str = "PX_CLOSE"
    ) -> pl.DataFrame:
        """Compute daily portfolio weights: date, ticker, weight.

        weight_i = shares_i * price_i / sum_j(shares_j * price_j)
        """
        filled = self._fill_holdings(price_data)
        if filled.is_empty():
            return pl.DataFrame(
                schema={"date": pl.Date, "ticker": pl.String, "weight": pl.Float64}
            )

        prices = price_data.select(
            [
                pl.col("ValueDate").alias("date"),
                pl.col("Ticker").alias("ticker"),
                pl.col(field).alias("price"),
            ]
        )

        joined = filled.join(prices, on=["date", "ticker"], how="left")
        joined = joined.with_columns((pl.col("shares") * pl.col("price")).alias("mv"))

        total_mv = joined.group_by("date").agg(pl.col("mv").sum().alias("total_mv"))
        joined = joined.join(total_mv, on="date")
        joined = joined.with_columns(
            (pl.col("mv") / pl.col("total_mv")).alias("weight")
        )

        return joined.select(["date", "ticker", "weight"]).sort(["date", "ticker"])

    def returns(
        self, price_data: pl.DataFrame, field: str = "PX_CLOSE"
    ) -> pl.DataFrame:
        """Weighted portfolio returns: {date, value}.

        Uses R_t = sum(w_{i,t-1} * r_{i,t}) where w_{i,t-1} is prior-day weight.
        """
        # Per-security returns, long format
        pr = (
            price_data.select(
                [
                    pl.col("ValueDate").alias("date"),
                    pl.col("Ticker").alias("ticker"),
                    pl.col(field).alias("price"),
                ]
            )
            .sort(["ticker", "date"])
            .with_columns(
                (pl.col("price") / pl.col("price").shift(1).over("ticker") - 1).alias(
                    "ret"
                )
            )
            .filter(pl.col("ret").is_not_null())
            .select(["date", "ticker", "ret"])
        )

        # Prior-period weights: shift within each ticker's series
        w = (
            self.weights(price_data, field)
            .sort(["ticker", "date"])
            .with_columns(pl.col("weight").shift(1).over("ticker").alias("prev_weight"))
            .filter(pl.col("prev_weight").is_not_null())
            .select(["date", "ticker", "prev_weight"])
        )

        joined = pr.join(w, on=["date", "ticker"], how="inner")

        return (
            joined.group_by("date")
            .agg((pl.col("ret") * pl.col("prev_weight")).sum().alias("value"))
            .sort("date")
        )

    def cumulative(
        self, price_data: pl.DataFrame, field: str = "PX_CLOSE"
    ) -> pl.DataFrame:
        """Cumulative return series indexed to 1.0."""
        ret = self.returns(price_data, field)
        return ret.with_columns(((pl.col("value") + 1).cum_prod()).alias("value"))

    def contribution(
        self, price_data: pl.DataFrame, field: str = "PX_CLOSE"
    ) -> pl.DataFrame:
        """Per-ticker contribution to portfolio returns: Ticker, date, value."""
        pr = (
            price_data.select(
                [
                    pl.col("ValueDate").alias("date"),
                    pl.col("Ticker").alias("ticker"),
                    pl.col(field).alias("price"),
                ]
            )
            .sort(["ticker", "date"])
            .with_columns(
                (pl.col("price") / pl.col("price").shift(1).over("ticker") - 1).alias(
                    "ret"
                )
            )
            .filter(pl.col("ret").is_not_null())
            .select(["date", "ticker", "ret"])
        )

        w = (
            self.weights(price_data, field)
            .sort(["ticker", "date"])
            .with_columns(pl.col("weight").shift(1).over("ticker").alias("prev_weight"))
            .filter(pl.col("prev_weight").is_not_null())
            .select(["date", "ticker", "prev_weight"])
        )

        joined = pr.join(w, on=["date", "ticker"], how="inner")
        return (
            joined.with_columns((pl.col("ret") * pl.col("prev_weight")).alias("value"))
            .rename({"ticker": "Ticker"})
            .select(["Ticker", "date", "value"])
        )

    def summary(self, price_data: pl.DataFrame, field: str = "PX_CLOSE") -> dict:
        """Key portfolio metrics.

        Returns dict with: total_return, annualized_return, volatility, sharpe,
        max_drawdown, calmar. Adds tracking_error, info_ratio, beta if benchmark
        is set and present in price_data.
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

    def to_portfolio(
        self, price_data: pl.DataFrame, dt: date, field: str = "PX_CLOSE"
    ) -> Portfolio:
        """Snapshot at a single date, returns a static Portfolio."""
        from qtk.portfolio import Portfolio

        w = self.weights(price_data, field)
        w_at_dt = w.filter(pl.col("date") == dt)

        if w_at_dt.is_empty():
            raise ValueError(f"No holdings data available at date {dt}")

        weights_dict = dict(
            zip(w_at_dt["ticker"].to_list(), w_at_dt["weight"].to_list())
        )
        return Portfolio(weights=weights_dict, benchmark=self.benchmark, name=self.name)

    @classmethod
    def from_dtk(
        cls,
        store,
        tickers: list[str],
        portfolio_id: str,
        start_dt: date | None = None,
        end_dt: date | None = None,
        name: str = "Portfolio",
        benchmark: str | None = None,
    ) -> PositionHistory:
        """Build from dtk Store position data."""
        pos = store.cds_position(
            tickers, portfolio_id=portfolio_id, start_dt=start_dt, end_dt=end_dt
        )
        if pos.is_empty():
            holdings = pl.DataFrame(
                schema={"date": pl.Date, "ticker": pl.String, "shares": pl.Float64}
            )
        else:
            holdings = pos.select(
                [
                    pl.col("ValueDate").alias("date"),
                    pl.col("Ticker").alias("ticker"),
                    pl.col("Shares").alias("shares"),
                ]
            )
        return cls(holdings=holdings, name=name, benchmark=benchmark)

    @classmethod
    def from_weights(
        cls,
        weights: dict[str, float],
        as_of: date | None = None,
        name: str = "Portfolio",
        benchmark: str | None = None,
    ) -> PositionHistory:
        """Build from a static weight dict (single-date holdings).

        Shares are set equal to weight values. Weights computed via the normal
        shares * price / total_mv formula will match the input weights only when
        all prices are equal; this factory is intended for API compatibility and
        testing with controlled price data.
        """
        if as_of is None:
            from datetime import date as _date

            as_of = _date.today()

        rows = [
            {"date": as_of, "ticker": t, "shares": float(w)} for t, w in weights.items()
        ]
        holdings = pl.DataFrame(
            rows,
            schema={"date": pl.Date, "ticker": pl.String, "shares": pl.Float64},
        )
        return cls(holdings=holdings, name=name, benchmark=benchmark)
