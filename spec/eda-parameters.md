# EDA Parameters (T1)

Companion to [cross-domain-eda.md](cross-domain-eda.md) — the concrete,
per-dataset constants `src/eda.py` is built against, decided here so code
and doc can't drift apart silently.

## Per-dataset unit conversion (display only; ADF/ACF are scale-invariant)

| Dataset | Native duration column | Native unit | Factor -> minutes |
|---|---|---|---|
| `synthetic_it_support_tickets` | `resolution_time_hours` | hours | ×60 |
| `call_center_emea_2021_2025` | `aHT_Seconds` | seconds | ÷60 |
| `generated_emergency_dept_stays` | `los_minutes` | minutes | ×1 |
| `generated_job_shop_manufacturing` | `processing_minutes` | minutes | ×1 |

## STL period and seasonal-boxplot grouping

Both taken directly from each dataset's existing `configs/<name>.yaml`
(`frequency`, `seasonal_period`) — not re-decided here.

| Dataset | `frequency` | STL `period` | Boxplot grouping |
|---|---|---|---|
| `synthetic_it_support_tickets` | D | 7 | day-of-week |
| `call_center_emea_2021_2025` | W | 52 | month-of-year |
| `generated_emergency_dept_stays` | D | 7 | day-of-week |
| `generated_job_shop_manufacturing` | D | 7 | day-of-week |

## Differencing / stationarity

- ADF significance level: **α = 0.05** (matches `quality-gate-thresholds.md`'s
  `adf_resid_p_max` convention already used in `recover_structure.py`).
- `suggested_d`: `0` if raw-series ADF p ≤ α; else `1` if first-differenced
  ADF p ≤ α; else the literal string `"d>=2 (not resolved)"` — per
  `cross-domain-eda.md`'s constraint, EDA does not auto-difference past d=1.

## Seasonality strength (STL-based)

Strength-of-seasonality measure (Hyndman & Athanasopoulos):
`F_s = max(0, 1 - Var(resid) / Var(seasonal + resid))`, computed from the
STL decomposition's `seasonal` and `resid` components.

- **Threshold: `F_s > 0.3` -> "visible seasonality: y"**, else "n".
- STL requires a complete (no-NaN) series. Each dataset already has
  `missing_pct <= 5%` (a passed Stage-0 gate condition), so any residual
  gaps are linearly interpolated **only for the STL input** — this
  interpolation is not applied to the series used for ADF/ACF/PACF or to
  the reported descriptive statistics, and is stated in each report.

## Dominant ACF/PACF lag

`nlags = min(40, n // 2 - 1)`. Dominant lag = the lag in `[1, nlags]` with
the largest `abs(acf value)` (reported for both ACF and PACF,
independently, on both the raw and first-differenced series).

## Metric list (per-dataset report + cross-domain summary — identical set)

`n`, `frequency`, `adf_p_d0`, `adf_p_d1`, `suggested_d`, `dominant_acf_lag`,
`dominant_pacf_lag`, `cv` (in minutes), `seasonal_strength` (`F_s`),
`visible_seasonality` (y/n), `has_ground_truth` (y/n).
