# EDA Interpretation — generated_emergency_dept_stays

Companion to [generated_emergency_dept_stays_eda.txt](generated_emergency_dept_stays_eda.txt)
and its five plots. **Ground truth exists** for this dataset
(`ground_truth.json`) — planted DGP: `m_t = 180 + 25·weekend +
20·cos(2π(doy−15)/365.25) + AR(1)(φ=0.60, σ=5)`, plus record-level
lognormal noise and an acuity-factor mixture. Where relevant, findings
below are cross-referenced against [reports/recovery_validation.txt](../recovery_validation.txt),
which fits the same series against the known planted values.

## Descriptive statistics

- **Record-level** (n=120,965 resolved encounters; 5,988 LWBS
  dropouts excluded): mean 205.1 min, sd 115.7, CV 0.56, median 176.0 min,
  p99 615.3 min, **skew 1.91** — strongly right-skewed, as expected from
  the planted acuity-factor mixture (factors from 0.55× to 2.2× the base
  level) combined with a clip ceiling at 1,440 min.
- **Series-level** (n=1,461, daily median): mean 177.1 min, sd 21.6, CV
  0.12, skew 0.36. Aggregation collapses most of the record-level skew and
  spread, as expected of a median over ~83 patients/day.

## Histogram (`_hist.png`)

A single right-skewed peak around 120–150 minutes with a long tail out
past 600 minutes — visually consistent with a population dominated by
mid-acuity patients (acuity 3–4, the two most probable levels) with a
smaller number of high-acuity, longer-stay patients (acuity 1–2) pulling
the tail out.

## Time series + rolling mean (`_series.png`)

A clear slow annual wave is visible in the rolling mean — peaks near each
January, troughs around mid-year — consistent with the planted
`20·cos(2π(doy−15)/365.25)` winter-surge term (phase reference day 15 ≈
mid-January). Daily noise sits on top of that wave, roughly 130–250 min.

## Seasonal boxplot by day-of-week (`_seasonal_boxplot.png`)

Saturday and Sunday medians (≈192–193 min) sit clearly above the five
weekdays (≈168–171 min) — a ~22–25 min weekend gap, closely matching the
planted **+25 min weekend effect**. This is the clearest weekly seasonal
signal of the four datasets.

## ACF / PACF (`_acf_pacf.png`) — read this one carefully

Raw series: ACF shows a slowly-decaying pattern with a repeating bump
every ~7 lags (matching the weekend effect) — dominant lag **7** (value
0.587). PACF's dominant lag is **1** (value 0.504). **This 0.504 is *not*
the same quantity as the planted AR(1) φ, and should not be read as "the
model recovered φ=0.50."** The raw-series PACF at lag 1 mixes together the
weekly/annual deterministic pattern *and* the latent AR(1) persistence,
because nothing has been removed from the series yet. `recover_structure.py`
does the OLS step first — regressing out the weekend dummy and annual
cosine — and only then fits AR(1)/ARMA(1,1) to what's left; that
residual-based fit is what recovers φ≈0.151 (see
`reports/recovery_validation.txt` and its `expected_recovery_note` in
`ground_truth.json`, which documents why record-level noise
(`lognormal_sigma=0.35`) plus the acuity mixture attenuates the recovered
AR coefficient this far below the planted 0.60). The two numbers (raw
PACF lag-1 ≈0.50 vs. residual-based φ≈0.15) are both correct — they're
just answering different questions, and conflating them would
overstate how much of this series' autocorrelation is "real" AR(1)
persistence versus deterministic seasonal structure.

First-differenced (d=1): PACF(d=1) shows a negative lag-1 value (≈−0.31)
followed by smaller negative values at lags 2–6, tapering toward 0 by
lag ~7 — this is consistent with **simple (non-seasonal) differencing
over-correcting a weekly-seasonal series**, and suggests seasonal
differencing (lag 7) or an explicit seasonal term may be more appropriate
than plain d=1 once actual model fitting starts. This is a diagnostic
observation only — order selection is out of scope for this stage.

## STL decomposition (`_stl.png`)

Trend: a clear repeating annual wave, roughly 150–225 min, matching the
planted annual cosine. Season: a tight, regular weekly ripple (roughly
±20–35 min) riding on top of the trend. Residual: the smallest relative to
signal of the four datasets (roughly ±50 min against a ~150–225 min
trend range). `seasonal_strength (F_s) = 0.522` — the **highest of the
four datasets**, correctly reflecting that this is the domain with the
strongest planted weekly seasonality.

## Stationarity (ADF)

d=0: p=0.0826 (non-stationary at α=0.05) — matches `recovery_validation.txt`'s
raw-series ADF exactly (0.0826), which notes this is the expected,
borderline result when deterministic seasonal/annual components are
planted. d=1: p=0.0000 (stationary). `suggested_d = 1`, though per the
PACF(d=1) note above, seasonal differencing may be a better fit than
simple differencing once modelling begins.

## What this suggests going into ARIMA fitting

- Clearest weekly seasonality of the four datasets (F_s=0.522,
  weekend-effect visible in every diagnostic) plus a slower annual cycle —
  this domain gives ARIMA the most seasonal structure to work with, but
  also the most risk of misspecification if seasonality isn't modelled
  explicitly (plain d=1 looks likely to under-perform a seasonal
  specification, per the PACF(d=1) pattern above).
- This is the domain already flagged (via `recovery_validation.txt`) as
  the one where record-level noise most heavily attenuates the recoverable
  AR signal — worth keeping in view for the RQ2 discussion of where ARIMA
  underperforms.
