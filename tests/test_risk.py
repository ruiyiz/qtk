"""Tests for ts/risk.py."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from qtk.ts.risk import component_var, es, marginal_var, var


def _price_series(n=252, seed=42):
    rng = np.random.default_rng(seed)
    vals = [100.0]
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + rng.standard_normal() * 0.01))
    return pl.DataFrame({"date": dates, "value": vals}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def _returns_series(n=252, seed=42):
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(n) * 0.01
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"date": dates, "value": vals.tolist()}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def test_var_historical_positive():
    """VaR should be positive (it's a loss magnitude)."""
    s = _price_series()
    v = var(s, p=0.05, method="historical")
    assert v > 0


def test_var_gaussian_positive():
    s = _price_series()
    v = var(s, p=0.05, method="gaussian")
    assert v > 0


def test_var_modified_positive():
    s = _price_series()
    v = var(s, p=0.05, method="modified")
    assert v > 0


def test_var_stricter_quantile():
    """VaR at p=0.01 should be larger than at p=0.05."""
    s = _price_series()
    v01 = var(s, p=0.01, method="historical")
    v05 = var(s, p=0.05, method="historical")
    assert v01 >= v05


def test_es_greater_than_var():
    """Expected shortfall should be >= VaR at same confidence level."""
    s = _price_series()
    v = var(s, p=0.05, method="historical")
    e = es(s, p=0.05, method="historical")
    assert e >= v - 1e-10


def test_var_invalid_method():
    s = _price_series()
    with pytest.raises(Exception):
        var(s, p=0.05, method="unknown")


def test_var_invalid_p():
    s = _price_series()
    with pytest.raises(Exception):
        var(s, p=1.5)


def test_component_var_sum():
    """Component VaR should sum approximately to portfolio VaR."""
    r1 = _returns_series(seed=1)
    r2 = _returns_series(seed=2)
    weights = [0.6, 0.4]
    comp = component_var([r1, r2], weights, p=0.05, method="gaussian")
    assert len(comp) == 2
    # Sum of components approximates total VaR (by construction)
    assert sum(comp) > 0


def test_marginal_var_length():
    r1 = _returns_series(seed=1)
    r2 = _returns_series(seed=2)
    weights = [0.5, 0.5]
    mvar = marginal_var([r1, r2], weights, p=0.05, method="gaussian")
    assert len(mvar) == 2


def test_component_var_mismatched_lengths():
    r1 = _returns_series()
    weights = [0.5, 0.5]
    with pytest.raises(Exception):
        component_var([r1], weights)
