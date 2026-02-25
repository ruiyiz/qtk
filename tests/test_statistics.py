import datetime as dt

import numpy as np
import polars as pl
import pytest

from qtk.ts.statistics import (
    Direction,
    LinearRegression,
    MeanType,
    RollingLinearRegression,
    cov,
    exponential_std,
    generate_series,
    max_,
    mean,
    median,
    min_,
    mode,
    percentile,
    percentiles,
    product,
    range_,
    std,
    sum_,
    var,
    winsorize,
    zscores,
)
from qtk.ts.helper import Window


def _s(vals, start=dt.date(2024, 1, 1)):
    dates = [start + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"date": dates, "value": [float(v) for v in vals]}).cast(
        {"date": pl.Date, "value": pl.Float64}
    )


def test_min_window():
    s = _s([3, 1, 4, 1, 5, 9, 2])
    result = min_(s, 3)
    assert result["value"][-1] == pytest.approx(2.0)


def test_max_window():
    s = _s([3, 1, 4, 1, 5, 9, 2])
    result = max_(s, 3)
    assert result["value"][-1] == pytest.approx(9.0)


def test_range_window():
    s = _s([1, 5, 3, 7, 2])
    result = range_(s, 3)
    # last window = [3, 7, 2], range = 7 - 2 = 5
    assert result["value"][-1] == pytest.approx(5.0)


def test_mean_window():
    s = _s([2, 4, 6, 8, 10])
    result = mean(s, 3)
    # last window = [6, 8, 10], mean = 8.0
    assert result["value"][-1] == pytest.approx(8.0)


def test_mean_quadratic():
    s = _s([3.0, 4.0])
    result = mean(s, Window(2, 0), MeanType.QUADRATIC)
    assert abs(result["value"][-1] - np.sqrt((9 + 16) / 2)) < 1e-10


def test_median_window():
    s = _s([1, 3, 2, 5, 4])
    result = median(s, 3)
    # last window = [2, 5, 4], median = 4.0
    assert result["value"][-1] == pytest.approx(4.0)


def test_sum_window():
    s = _s([1, 2, 3, 4])
    result = sum_(s, 2)
    assert result["value"][-1] == pytest.approx(7.0)


def test_product_window():
    s = _s([1, 2, 3, 4, 5])
    result = product(s, 3)
    # last window = [3, 4, 5], product = 60
    assert result["value"][-1] == pytest.approx(60.0)


def test_std_window():
    vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    s = _s(vals)
    expected = np.std(vals, ddof=1)
    result = std(s, Window(len(vals), 0))
    assert abs(result["value"][-1] - expected) < 1e-6


def test_var_window():
    vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    s = _s(vals)
    expected = np.var(vals, ddof=1)
    result = var(s, Window(len(vals), 0))
    assert abs(result["value"][-1] - expected) < 1e-6


def test_exponential_std():
    s = _s([1.0, 2.0, 3.0, 4.0, 5.0])
    result = exponential_std(s, beta=0.9)
    assert result.height == s.height
    assert result["value"][-1] > 0


def test_cov_identical():
    s = _s([1.0, 2.0, 3.0, 4.0, 5.0])
    result = cov(s, s, Window(5, 0))
    # Cov(X, X) = Var(X)
    v = var(s, Window(5, 0))
    assert abs(result["value"][-1] - v["value"][-1]) < 1e-10


def test_zscores_full():
    s = _s([1.0, 2.0, 3.0, 4.0, 5.0])
    result = zscores(s)
    # z-score of last value (5.0) in sample
    vals = s["value"].to_numpy()
    expected_z = (5.0 - vals.mean()) / vals.std(ddof=1)
    assert abs(result["value"][-1] - expected_z) < 1e-6


def test_winsorize_clips():
    # Need enough tight points so outlier falls outside mean + 2*sigma
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10000.0]
    s = _s(vals)
    result = winsorize(s, limit=2.0)
    # The extreme value 10000 should be clipped below the original
    assert result["value"][-1] < 10000.0


def test_percentile_scalar():
    s = _s([1.0, 2.0, 3.0, 4.0, 5.0])
    p = percentile(s, 50)
    assert p == pytest.approx(3.0)


def test_percentile_rolling():
    s = _s([1.0, 2.0, 3.0, 4.0, 5.0])
    result = percentile(s, 50, 3)
    # normalize_window(s, 3) → Window(3, 3); apply_ramp slices off first 3 rows → 2 remain
    assert result.height == 2


def test_percentiles_self():
    s = _s([10.0, 20.0, 30.0, 40.0, 50.0])
    result = percentiles(s, None, Window(5, 0))
    # kind='mean': percentile of max = mean(80%, 100%) = 90%
    assert result["value"][-1] == pytest.approx(90.0)


def test_generate_series():
    result = generate_series(50, Direction.START_TODAY)
    assert result.height == 50
    assert result["value"][0] == pytest.approx(100.0)


def test_linear_regression():
    x = _s([1.0, 2.0, 3.0, 4.0, 5.0])
    y = _s([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x
    reg = LinearRegression(x, y, fit_intercept=True)
    # R^2 should be 1.0 for perfect fit
    assert reg.r_squared() == pytest.approx(1.0, abs=1e-6)
    # Slope coefficient (index 1) should be ~2
    assert abs(reg.coefficient(1) - 2.0) < 0.01


def test_rolling_linear_regression():
    x = _s([float(i) for i in range(1, 21)])
    y = _s([float(i) * 2 for i in range(1, 21)])
    rlr = RollingLinearRegression(x, y, w=5, fit_intercept=True)
    r2 = rlr.r_squared()
    assert r2.height > 0
    # R^2 should be 1.0 for perfect linear relationship
    non_null = r2.drop_nulls("value")
    assert all(abs(v - 1.0) < 1e-4 for v in non_null["value"].to_list())
