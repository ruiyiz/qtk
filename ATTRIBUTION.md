# Attribution

## Goldman Sachs gs-quant

Portions of this project are derived from [gs-quant](https://github.com/goldmansachs/gs-quant),
Copyright 2018 Goldman Sachs, licensed under the Apache License, Version 2.0.

See `licenses/Apache-2.0-gs-quant.txt` for the full license text.
See `licenses/gs-quant-NOTICE` and `licenses/gs-quant-NOTICE.txt` for the original NOTICE files.

### Extracted source files

| gs-quant source | qtk target | Notes |
|---|---|---|
| `gs_quant/errors.py` | `src/qtk/errors.py` | Error hierarchy |
| `gs_quant/datetime/date.py` | `src/qtk/date_utils.py` | Day count conventions (GsCalendar functions removed) |
| `gs_quant/timeseries/helper.py` | `src/qtk/ts/helper.py` | Window, enums, normalize_window, apply_ramp |
| `gs_quant/timeseries/datetime.py` | `src/qtk/ts/dateops.py` | align, interpolate, bucketize, etc. (align_calendar removed) |
| `gs_quant/timeseries/algebra.py` | `src/qtk/ts/algebra.py` | All 20 algebraic operations |
| `gs_quant/timeseries/analysis.py` | `src/qtk/ts/analysis.py` | lag, diff, compare, smooth_spikes, etc. |
| `gs_quant/timeseries/statistics.py` | `src/qtk/ts/statistics.py` | Rolling stats, regression (SIRModel/SEIRModel removed) |
| `gs_quant/timeseries/econometrics.py` | `src/qtk/ts/econometrics.py` | returns, volatility, correlation, beta, etc. (Marquee functions removed) |
| `gs_quant/timeseries/technicals.py` | `src/qtk/ts/technicals.py` | MA, RSI, MACD, seasonal decomposition |

### Excluded functions (Marquee API dependency)

- `align_calendar` -- depends on `GsCalendar`
- `excess_returns_` -- always Marquee
- `sharpe_ratio` -- always Marquee
- `SIRModel`, `SEIRModel` -- depend on `lmfit` epidemiology module
- `is_business_day`, `business_day_offset`, `business_day_count`, `date_range`,
  `prev_business_date`, `today` -- all depend on `GsCalendar`

### Modifications applied

1. **Pandas to Polars**: All `pd.Series` inputs/outputs replaced with `pl.DataFrame`
   with schema `{"date": pl.Date, "value": pl.Float64}`.
2. **NumPy 2.0 compatibility**: `np.double` replaced with `np.float64` throughout.
3. **Tenor string mapping**: gs-quant `"1m"` (month) mapped to polars `"1mo"` via
   `_to_polars_duration()` (polars uses `"m"` for minutes).
4. **EWM**: `series.ewm(alpha=a, adjust=False).mean()` mapped to
   `pl.col("value").ewm_mean(alpha=a, adjust=False)`.
5. **Rolling correlation/covariance**: Implemented using `numpy.corrcoef` /
   `numpy.cov` over each window (no native polars equivalent).
6. **Seasonal decomposition**: `seasonally_adjusted` and `trend` convert internally
   to pandas for `statsmodels.tsa.seasonal.seasonal_decompose`.
7. **Linear regression**: `LinearRegression` and `RollingLinearRegression` convert
   inputs to pandas/numpy internally for statsmodels calls.
8. **Error mapping**: `MqValueError` -> `QtkValueError`, `MqTypeError` -> `QtkTypeError`,
   `MqError` -> `QtkError`.

### Mathematical preservation

All mathematical formulae and algorithms are unchanged from the original gs-quant source.
The modifications are purely mechanical: API layer (pandas->polars), error types
(Mq*->Qtk*), and deprecated NumPy symbol names (np.double->np.float64).
