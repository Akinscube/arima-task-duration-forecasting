# EDA Interpretation — generated_job_shop_manufacturing

Companion to [generated_job_shop_manufacturing_eda.txt](generated_job_shop_manufacturing_eda.txt)
and its five plots. **Ground truth exists** for this dataset
(`ground_truth.json`) — planted DGP: `p_t = 55 + ARMA(1,1)(φ=0.85,
θ=0.35, σ=2.0)`, no seasonality planted, tight per-record noise
(`lognormal_sigma=0.04`). Cross-referenced against
[reports/recovery_validation.txt](../recovery_validation.txt) where noted.

## Descriptive statistics

- **Record-level** (n=65,911 jobs, no dropout): mean 55.16 min, sd 5.32,
  **CV 0.096 — the lowest of the four datasets**, skew 0.19 (nearly
  symmetric).
- **Series-level** (n=1,461, daily median): mean 55.13 min, sd 4.86, CV
  0.088, skew 0.15.

Unlike the other three datasets, record-level and series-level statistics
are almost identical here (CV 0.096 → 0.088, a small drop). Daily-median
aggregation barely changes the distribution's shape or spread. That's
the direct consequence of the DGP design: per-record noise is tight
(`lognormal_sigma=0.04`, versus 0.35 for the ED dataset), so aggregation
has little noise left to average out — most of the series-level variance
is the genuine ARMA(1,1) signal (φ=0.85 is high persistence), not
record-level noise.

## Histogram (`_hist.png`)

A clean, roughly symmetric, unimodal bell shape centered at ~55 minutes,
range ~40–75 minutes. The most "textbook normal-looking" distribution of
the four datasets — consistent with tight engineered tolerances and no
acuity-style multiplicative mixture.

## Time series + rolling mean (`_series.png`)

Visibly smoother and more wave-like than a noise-around-a-flat-mean
series — the rolling mean itself oscillates between roughly 48 and 65
minutes over multi-week spans. This is the ARMA(1,1) persistence itself
showing through: φ=0.85 means today's level strongly resembles
yesterday's, producing sustained multi-day swings rather than day-to-day
noise. No planted seasonality, and none is visually apparent.

## Seasonal boxplot by day-of-week (`_seasonal_boxplot.png`)

All seven days show near-identical medians (≈54.7–55.3 min) and heavily
overlapping IQRs — no day-of-week effect, as expected: this dataset's
`ground_truth.json` entry explicitly states "seasonality: none planted
(persistence-dominated domain)." The EDA correctly finds nothing here,
which is itself a useful check — it shows the pipeline doesn't manufacture
a seasonal signal where none exists.

## ACF / PACF (`_acf_pacf.png`)

Raw series: ACF decays slowly and smoothly from 0.90 at lag 1 down toward
0 by around lag 28, going slightly negative beyond that — the classic
signature of a highly persistent AR/ARMA process, not of seasonality
(there's no repeating bump pattern the way the ED dataset shows at
multiples of 7). PACF cuts off sharply after lag 1 (0.90), but not
perfectly — there's a distinct negative value at lag 2 (≈−0.23) before
dropping into the noise band. A pure AR(1) process would cut off cleanly
after lag 1; this small but real lag-2 PACF value is the fingerprint of
the planted **θ=0.35 MA(1) component** riding alongside the AR(1) term.
First-differenced (d=1): ACF/PACF drop to small values (lag 1 ≈0.17,
lag 2 ≈ −0.12–0.16) — much closer to noise than the raw series, which is
expected: differencing an already-stationary, highly persistent series
removes real signal rather than a unit root, another over-differencing
case. `suggested_d = 0` (raw ADF p≈0.0000) is clearly the right call.

## STL decomposition (`_stl.png`)

Trend: tracks the same multi-week undulations visible in the raw series
plot (roughly 50–65 min) — but this is **not a genuine long-run drift**,
it's STL's low-frequency smoother absorbing the ARMA(1,1) process's own
persistence, since there's no repeating weekly pattern for STL to route
into "season" instead. Season: small, roughly ±3–5 min, essentially
noise. Residual: small and tight, roughly ±10 min. `seasonal_strength
(F_s) = 0.096` — the **lowest of the four datasets**, correctly reflecting
the absence of planted seasonality.

## Cross-check against recovery_validation.txt

`recover_structure.py` fits ARMA(1,1) directly to this series (no
deterministic terms to remove, since none are planted) and recovers
φ 0.85→0.849, θ 0.35→0.298 — close to the planted values, with AIC
correctly preferring ARMA(1,1) over AR(1). This EDA's raw-series PACF
lag-1 value (0.902) is close to but slightly above that residual-based
recovered φ (0.849), which makes sense: unlike the ED dataset, there's no
deterministic seasonal component here to confound the raw PACF with, so
the two measurements agree much more closely than they do for
`generated_emergency_dept_stays`.

## What this suggests going into ARIMA fitting

- The most "ARIMA-friendly" domain of the four: strong, genuine
  autocorrelation (not seasonality-driven), no seasonal complications, and
  `d=0` is unambiguous.
- Best candidate for demonstrating ARIMA's advantage over baselines, since
  its persistence (φ=0.85) is exactly the structure a naïve or moving-
  average baseline can't exploit as well as a properly-specified
  ARMA(1,1)/ARIMA(1,0,1).
