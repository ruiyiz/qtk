# gs-quant Timeseries Port: Audit

## Function Classification

### STANDALONE (no Marquee dependency, fully ported)

| Function | Module | Notes |
|---|---|---|
| `smooth_spikes` | analysis | |
| `repeat` | analysis | |
| `first`, `last`, `last_value` | analysis | |
| `count`, `diff`, `compare`, `lag` | analysis | |
| `min_`, `max_`, `range_` | statistics | |
| `mean`, `median`, `mode` | statistics | |
| `sum_`, `product` | statistics | |
| `std`, `exponential_std`, `var`, `cov` | statistics | |
| `zscores`, `winsorize` | statistics | |
| `generate_series`, `generate_series_intraday` | statistics | |
| `percentiles`, `percentile` | statistics | |
| `LinearRegression`, `RollingLinearRegression` | statistics | uses statsmodels internally |
| `returns`, `prices`, `index`, `change` | econometrics | |
| `annualize`, `volatility` | econometrics | |
| `vol_swap_volatility` | econometrics | |
| `correlation`, `corr_swap_correlation` | econometrics | |
| `beta`, `max_drawdown` | econometrics | |
| `excess_returns_pure` | econometrics | two-series version, no API |
| `excess_returns` (float path) | econometrics | float rate only |
| `get_ratio_pure` | econometrics | |
| `moving_average`, `bollinger_bands` | technicals | |
| `smoothed_moving_average` | technicals | |
| `relative_strength_index` | technicals | |
| `exponential_moving_average`, `macd` | technicals | |
| `exponential_volatility`, `exponential_spread_volatility` | technicals | |
| `seasonally_adjusted`, `trend` | technicals | uses pandas/statsmodels internally |
| `align`, `interpolate`, `value` | dateops | |
| `day`, `month`, `year`, `quarter`, `weekday` | dateops | |
| `day_count_fractions`, `date_range` | dateops | |
| `append`, `prepend`, `union` | dateops | |
| `bucketize`, `day_count`, `day_countdown` | dateops | |
| `add`, `subtract`, `multiply`, `divide`, `floordiv` | algebra | |
| `exp`, `log`, `power`, `sqrt`, `abs_`, `floor`, `ceil` | algebra | |
| `filter_`, `filter_dates` | algebra | |
| `and_`, `or_`, `not_`, `if_` | algebra | |
| `weighted_sum`, `geometrically_aggregate` | algebra | |
| `day_count_fraction`, `has_feb_29` | date_utils | |

### EXCLUDED (Marquee API dependency)

| Function | Reason |
|---|---|
| `excess_returns_` | Always calls Marquee GsDataApi |
| `sharpe_ratio` | Always calls Marquee GsDataApi |
| `SIRModel`, `SEIRModel` | Depend on `lmfit` epidemiology module |
| `align_calendar` | Depends on `GsCalendar.get()` |
| `is_business_day`, `business_day_offset`, `business_day_count` | Depend on `GsCalendar` |
| `date_range` (gs_quant/datetime) | Depends on `GsCalendar` |
| `prev_business_date`, `today` | Depend on `GsCalendar` |

---

## Pandas to Polars Translation Guide

| pandas pattern | polars equivalent |
|---|---|
| `pd.Series(data, index=dates)` | `pl.DataFrame({"date": dates, "value": data})` |
| `s.rolling(w, 0).mean()` | `pl.col("value").rolling_mean(window_size=w, min_periods=1)` |
| `s.rolling(w, 0).std()` | `pl.col("value").rolling_std(window_size=w, min_periods=1)` |
| `s.rolling(w, 0).min()` | `pl.col("value").rolling_min(window_size=w, min_periods=1)` |
| `s.rolling(w, 0).max()` | `pl.col("value").rolling_max(window_size=w, min_periods=1)` |
| `s.rolling(w, 0).sum()` | `pl.col("value").rolling_sum(window_size=w, min_periods=1)` |
| `s.ewm(alpha=a, adjust=False).mean()` | `pl.col("value").ewm_mean(alpha=a, adjust=False)` |
| `s.ewm(alpha=a, adjust=False).std()` | `pl.col("value").ewm_std(alpha=a, adjust=False)` |
| `s.shift(n)` | `pl.col("value").shift(n)` |
| `s.cumsum()` | `pl.col("value").cum_sum()` |
| `s.cumprod()` | `pl.col("value").cum_prod()` |
| `s.apply(math.log)` | `pl.col("value").log(base=math.e)` |
| `s.apply(math.exp)` | `pl.col("value").exp()` |
| `pd.concat([a, b], axis=1)` | `a.join(b, on="date", ...)` |
| `x.align(y, 'inner', axis=0)` | `x.join(y, on="date", how="inner")` |
| `s.rolling(w).corr(other)` | custom numpy per-window loop |
| `s.rolling(w).cov(other)` | custom numpy per-window loop |
| `group_by_dynamic` | `pl.DataFrame.group_by_dynamic(...)` |

---

## NumPy 2.0 Patches Applied

| Deprecated symbol | Replacement | Location |
|---|---|---|
| `np.double` | `np.float64` | statistics.py (rolling_std helper) |

No other deprecated `np.bool`, `np.int`, `np.float`, `np.complex`, `np.object`,
`np.str`, `np.NaN`, `np.Inf` patterns were found in the ported source files.

---

## Key Design Decisions

### Return type
All ported time series functions return `pl.DataFrame` with columns:
- `date` (`pl.Date`)
- `value` (`pl.Float64`)

Exception: `generate_series_intraday` returns `{"datetime": pl.Datetime, "value": pl.Float64}`.
Exception: `bollinger_bands` returns `{"date": pl.Date, "lower": pl.Float64, "upper": pl.Float64}`.

### Rolling windows
- **Int windows**: `pl.col("value").rolling_*(window_size=w, min_periods=1)`
- **Duration windows**: custom numpy loop using `_to_timedelta(w.w)` for window boundary

### Tenor strings
gs-quant `"1m"` (month) maps to polars `"1mo"` (polars uses `"m"` for minutes).
Conversion via `helper._to_polars_duration()`.

### align() implementation
Uses polars joins:
- `INTERSECT` -- inner join on `date`
- `NAN` -- full join, nulls preserved
- `ZERO` -- full join, fill nulls with 0
- `STEP` -- full join, forward_fill then backward_fill
- `TIME` -- full join, interpolate_by date (cast to Int32)

### statsmodels interop
`LinearRegression`, `RollingLinearRegression`, `seasonally_adjusted`, `trend`
convert polars DataFrames to pandas/numpy internally for statsmodels calls,
then convert results back to polars DataFrames.
