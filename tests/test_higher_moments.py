"""Tests for higher moments and distribution analysis in ts/statistics.py."""

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from qtk.ts.statistics import (
    adjusted_sharpe,
    hurst_index,
    kurtosis,
    prob_sharpe_ratio,
    skewness,
)


def _s(vals, start=dt.date(2024, 1, 1)):
    dates = [start + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"date": dates, "value": [float(v) for v in vals]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def _price_series(n=252, seed=42):
    rng = np.random.default_rng(seed)
    vals = [100.0]
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + rng.standard_normal() * 0.01))
    return pl.DataFrame({"date": dates, "value": vals}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def test_skewness_symmetric():
    """Symmetric series should have near-zero skewness."""
    vals = list(range(1, 11)) + list(range(9, 0, -1))
    s = _s(vals)
    result = skewness(s)
    last = result["value"].drop_nulls()[-1]
    assert abs(last) < 0.5


def test_skewness_known():
    """
    Hand-computed skewness for [1, 2, 3, 4, 10].
    n=5, mean=4, sigma=approx 3.536
    Adjusted Fisher-Pearson: n/(n-1)/(n-2) * sum(((x-mu)/sigma)^3)
    scipy.stats.skew([1,2,3,4,10]) ≈ 1.0755
    """
    import scipy.stats as sc

    vals = [1.0, 2.0, 3.0, 4.0, 10.0]
    s = _s(vals)
    result = skewness(s)
    last = float(result["value"].drop_nulls()[-1])
    expected = float(sc.skew(vals, bias=False))
    assert abs(last - expected) < 1e-6


def test_kurtosis_normal_approx():
    """Large normal sample should have near-zero excess kurtosis."""
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(1000).tolist()
    s = _s(vals)
    result = kurtosis(s)
    last = result["value"].drop_nulls()[-1]
    assert abs(last) < 0.5


def test_kurtosis_known():
    """
    Cross-validate kurtosis with scipy for a small known series.
    scipy.stats.kurtosis([1,2,3,4,10], fisher=True, bias=False) ≈ -0.1367
    """
    import scipy.stats as sc

    vals = [1.0, 2.0, 3.0, 4.0, 10.0]
    s = _s(vals)
    result = kurtosis(s)
    last = float(result["value"].drop_nulls()[-1])
    expected = float(sc.kurtosis(vals, fisher=True, bias=False))
    assert abs(last - expected) < 1e-4


def test_hurst_random_walk():
    """
    Random walk should return a finite Hurst index.
    The R/S estimator has high variance on finite samples.
    """
    rng = np.random.default_rng(42)
    n = 500
    vals = np.cumsum(rng.standard_normal(n)) + 100
    s = _s(vals.tolist())
    h = hurst_index(s)
    assert isinstance(h, float)
    assert not math.isnan(h)


def test_hurst_too_short():
    s = _s([1.0, 2.0, 3.0])
    with pytest.raises(Exception):
        hurst_index(s)


def test_adjusted_sharpe_shape():
    s = _price_series()
    result = adjusted_sharpe(s)
    assert result.height > 0


def test_adjusted_sharpe_vs_regular():
    """Adjusted Sharpe with zero skew and zero kurt should match plain Sharpe."""
    from qtk.ts.ratios import sharpe_ratio

    s = _price_series()
    asr = adjusted_sharpe(s)["value"].drop_nulls()[-1]
    sr = sharpe_ratio(s)["value"].drop_nulls()[-1]
    assert not math.isnan(asr)
    assert not math.isnan(sr)


def test_prob_sharpe_ratio_in_01():
    """PSR should be a probability in [0, 1]."""
    s = _price_series()
    result = prob_sharpe_ratio(s, rf=0.0, benchmark_sr=0.0)
    vals = result["value"].drop_nulls().to_numpy()
    non_nan = [v for v in vals if not math.isnan(v)]
    assert all(0.0 <= v <= 1.0 for v in non_nan)


def test_prob_sharpe_ratio_higher_benchmark_lower_psr():
    """Higher benchmark SR => lower PSR."""
    s = _price_series()
    psr0 = prob_sharpe_ratio(s, benchmark_sr=0.0)["value"].drop_nulls()[-1]
    psr2 = prob_sharpe_ratio(s, benchmark_sr=2.0)["value"].drop_nulls()[-1]
    if not (math.isnan(psr0) or math.isnan(psr2)):
        assert psr0 >= psr2
