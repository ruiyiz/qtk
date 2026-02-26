"""
Value at Risk and Expected Shortfall: historical, Gaussian, and Cornish-Fisher methods.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import numpy as np
import polars as pl

from qtk.errors import QtkValueError
from qtk.ts.econometrics import returns as _returns

__all__ = [
    "var",
    "es",
    "component_var",
    "marginal_var",
]

_VALID_METHODS = ("historical", "gaussian", "modified")


def _check_p(p: float) -> None:
    if not 0 < p < 1:
        raise QtkValueError("p must be in (0, 1)")


def _cornish_fisher_z(p: float, skew: float, kurt: float) -> float:
    """Adjusted z-score using the Cornish-Fisher expansion."""
    z = _norm_ppf(p)
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )
    return z_cf


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via scipy."""
    from scipy.stats import norm

    return float(norm.ppf(p))


def var(
    x: pl.DataFrame,
    p: float = 0.05,
    method: str = "historical",
) -> float:
    """
    Value at Risk (loss threshold at probability p).

    :param x: price timeseries
    :param p: tail probability (e.g. 0.05 = 5% VaR)
    :param method: 'historical', 'gaussian', or 'modified' (Cornish-Fisher)
    :return: VaR as a positive fraction (e.g. 0.02 means 2% loss)
    """
    _check_p(p)
    if method not in _VALID_METHODS:
        raise QtkValueError(f"method must be one of {_VALID_METHODS}")

    ret = _returns(x)["value"].drop_nulls().to_numpy()

    if method == "historical":
        return float(-np.quantile(ret, p))

    mu = float(np.mean(ret))
    sigma = float(np.std(ret, ddof=1))

    if method == "gaussian":
        return float(-(mu + sigma * _norm_ppf(p)))

    skew = float(_sample_skewness(ret))
    kurt = float(_sample_excess_kurtosis(ret))
    z_cf = _cornish_fisher_z(p, skew, kurt)
    return float(-(mu + sigma * z_cf))


def es(
    x: pl.DataFrame,
    p: float = 0.05,
    method: str = "historical",
) -> float:
    """
    Expected Shortfall (CVaR): mean loss conditional on exceeding VaR.

    :param x: price timeseries
    :param p: tail probability (e.g. 0.05 = 5% ES)
    :param method: 'historical', 'gaussian', or 'modified' (Cornish-Fisher)
    :return: ES as a positive fraction
    """
    _check_p(p)
    if method not in _VALID_METHODS:
        raise QtkValueError(f"method must be one of {_VALID_METHODS}")

    ret = _returns(x)["value"].drop_nulls().to_numpy()
    var_val = var(x, p, method)

    if method == "historical":
        tail = ret[ret <= -var_val]
        return float(-np.mean(tail)) if len(tail) > 0 else var_val

    mu = float(np.mean(ret))
    sigma = float(np.std(ret, ddof=1))

    from scipy.stats import norm

    if method == "gaussian":
        return float(-(mu - sigma * norm.pdf(norm.ppf(p)) / p))

    skew = float(_sample_skewness(ret))
    kurt = float(_sample_excess_kurtosis(ret))
    z_cf = _cornish_fisher_z(p, skew, kurt)
    # Use the Cornish-Fisher adjusted VaR as basis for ES approximation
    # (integrate numerically over the tail region)
    z_threshold = _norm_ppf(p)
    from scipy.integrate import quad

    def integrand(z):
        z_adj = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )
        return (mu + sigma * z_adj) * norm.pdf(z)

    integral, _ = quad(integrand, -np.inf, z_threshold)
    return float(-integral / p)


def component_var(
    returns_list: List[pl.DataFrame],
    weights: List[float],
    p: float = 0.05,
    method: str = "historical",
) -> List[float]:
    """
    Per-asset component VaR contribution to portfolio VaR.

    :param returns_list: list of return timeseries (each already a returns series, not prices)
    :param weights: portfolio weights (sum to 1)
    :param p: tail probability
    :param method: 'historical', 'gaussian', or 'modified'
    :return: list of component VaR values (sum equals portfolio VaR)
    """
    _check_p(p)
    if len(returns_list) != len(weights):
        raise QtkValueError("returns_list and weights must have the same length")

    weights_arr = np.array(weights, dtype=float)
    port_ret = _portfolio_returns(returns_list, weights_arr)

    cov_matrix = _cov_matrix(returns_list)
    port_std = math.sqrt(float(weights_arr @ cov_matrix @ weights_arr))

    from scipy.stats import norm

    if method == "historical":
        z = -np.quantile(port_ret, p)
    elif method == "gaussian":
        z = port_std * (-_norm_ppf(p))
    else:
        mu = float(np.mean(port_ret))
        sigma = float(np.std(port_ret, ddof=1))
        skew = float(_sample_skewness(port_ret))
        kurt = float(_sample_excess_kurtosis(port_ret))
        z_cf = _cornish_fisher_z(p, skew, kurt)
        z = -(mu + sigma * z_cf)

    marginal = (
        cov_matrix @ weights_arr / port_std
        if port_std > 0
        else np.zeros_like(weights_arr)
    )
    component = (
        weights_arr * marginal * (z / port_std)
        if port_std > 0
        else np.zeros_like(weights_arr)
    )
    return component.tolist()


def marginal_var(
    returns_list: List[pl.DataFrame],
    weights: List[float],
    p: float = 0.05,
    method: str = "historical",
) -> List[float]:
    """
    Marginal VaR: change in portfolio VaR per unit increase in each asset weight.

    :param returns_list: list of return timeseries
    :param weights: portfolio weights (sum to 1)
    :param p: tail probability
    :param method: 'historical', 'gaussian', or 'modified'
    :return: list of marginal VaR values
    """
    _check_p(p)
    if len(returns_list) != len(weights):
        raise QtkValueError("returns_list and weights must have the same length")

    weights_arr = np.array(weights, dtype=float)
    port_ret = _portfolio_returns(returns_list, weights_arr)

    cov_matrix = _cov_matrix(returns_list)
    port_std = math.sqrt(float(weights_arr @ cov_matrix @ weights_arr))

    if method == "historical":
        z_scalar = -np.quantile(port_ret, p) / port_std if port_std > 0 else 0.0
    elif method == "gaussian":
        z_scalar = -_norm_ppf(p)
    else:
        mu = float(np.mean(port_ret))
        sigma = float(np.std(port_ret, ddof=1))
        skew = float(_sample_skewness(port_ret))
        kurt = float(_sample_excess_kurtosis(port_ret))
        z_cf = _cornish_fisher_z(p, skew, kurt)
        z_scalar = -(mu + sigma * z_cf) / port_std if port_std > 0 else 0.0

    marginal = (
        cov_matrix @ weights_arr / port_std * z_scalar
        if port_std > 0
        else np.zeros_like(weights_arr)
    )
    return marginal.tolist()


def _portfolio_returns(
    returns_list: List[pl.DataFrame], weights: np.ndarray
) -> np.ndarray:
    """Compute weighted portfolio returns aligned on common dates."""
    joined = returns_list[0].rename({"value": "r0"})
    for i, df in enumerate(returns_list[1:], start=1):
        joined = joined.join(df.rename({"value": f"r{i}"}), on="date", how="inner")
    mat = joined.select([c for c in joined.columns if c != "date"]).to_numpy()
    return mat @ weights


def _cov_matrix(returns_list: List[pl.DataFrame]) -> np.ndarray:
    """Compute covariance matrix from a list of return DataFrames."""
    joined = returns_list[0].rename({"value": "r0"})
    for i, df in enumerate(returns_list[1:], start=1):
        joined = joined.join(df.rename({"value": f"r{i}"}), on="date", how="inner")
    mat = joined.select([c for c in joined.columns if c != "date"]).to_numpy()
    return np.cov(mat.T, ddof=1)


def _sample_skewness(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 3:
        return 0.0
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if sigma == 0:
        return 0.0
    return float(n / ((n - 1) * (n - 2)) * np.sum(((arr - mu) / sigma) ** 3))


def _sample_excess_kurtosis(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 4:
        return 0.0
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if sigma == 0:
        return 0.0
    raw = float(
        n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(((arr - mu) / sigma) ** 4)
        - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    )
    return raw
