# EDA Interpretation — synthetic_it_support_tickets

Companion to [synthetic_it_support_tickets_eda.txt](synthetic_it_support_tickets_eda.txt)
and its five plots. No ground truth exists for this dataset (third-party
Kaggle synthetic data, per `CLAUDE.md`) — findings below are exploratory,
not validated against a known generating process.

## Descriptive statistics

- **Record-level** (n=60,113 resolved tickets; 39,887 unresolved/NaN
  tickets dropped before this stage): mean 2,700.8 min (≈45.0 h), sd
  2,952.2, **CV 1.09** — the standard deviation exceeds the mean, the only
  dataset of the four where that happens. Median 1,794.6 min (≈29.9 h),
  p99 13,591.8 min (≈9.4 days), **skew 2.18** — strongly right-skewed.
- **Series-level** (n=1,460, daily median): mean 1,804.6 min, sd 355.9,
  CV 0.20, skew 0.16.

The gap between record-level (CV 1.09, skew 2.18) and series-level (CV
0.20, skew 0.16) is the largest of the four datasets — daily-median
aggregation is doing a lot of work here, averaging out a genuinely
heavy-tailed per-ticket distribution into a much more modest daily signal.

## Histogram (`_hist.png`)

A tall peak around 500–1,500 minutes, then a step down around 4,300–4,500
minutes into a long, low, roughly flat "shelf" running out to ~14,000
minutes. That shelf shape — not a smoothly decaying tail — suggests two
overlapping populations of tickets rather than one smooth distribution:
most tickets resolve within a few days, but a distinct, roughly-uniform
population of tickets remains open for anywhere from ~3 to ~10 days
(possibly a different priority tier or an SLA-driven escalation queue).
This is a descriptive observation from the shape, not something the data
confirms directly — worth checking against any priority/category column
if one exists in the raw file.

## Time series + rolling mean (`_series.png`)

Noisy day-to-day (roughly 1,000–3,000 min), oscillating around a stable
rolling mean near 1,750–1,900 min for the full four years — no visible
trend, no visible level shift. Visually the closest of the four datasets
to a stationary, structure-free series.

## Seasonal boxplot by day-of-week (`_seasonal_boxplot.png`)

All seven days show near-identical medians (≈1,750–1,870 min) and
overlapping IQRs — no day stands out. Consistent with the STL result
below.

## ACF / PACF (`_acf_pacf.png`)

Raw series: every lag from 1 to 40 sits inside or barely brushes the 95%
confidence band. The reported "dominant" lag (35, value −0.071) is only
dominant in a relative sense — in absolute terms it's negligible. This is
the **weakest temporal structure of the four datasets** — the series
behaves close to white noise around a stable level. First-differenced
(d=1): a sharp negative spike at lag 1 (≈−0.49), again the standard
over-differencing signature, confirming `suggested_d = 0` is correct — the
raw series is already stationary (ADF d=0 p≈0.0000) and has essentially no
autocorrelation left to remove.

## STL decomposition (`_stl.png`)

Trend: fairly flat, oscillating mildly around 1,750–2,000 min. Season: a
visible repeating weekly ripple (~±500 min amplitude) — it *is* there
visually. Residual: much larger, ±1,000+ min with occasional bigger
spikes. `seasonal_strength (F_s) = 0.221`, below the 0.3 threshold, hence
`visible_seasonality: n` — the seasonal ripple exists but is small relative
to the dominant residual noise, which is the correct read given how flat
the day-of-week boxplot is.

## What this suggests going into ARIMA fitting

- This is the domain with the least exploitable autocorrelation of the
  four — almost everything in the series looks like noise around a
  constant level, once the heavy per-ticket tail is averaged away by daily
  aggregation.
- `d=0` is clearly correct; no seasonal differencing is indicated either.
- For RQ2: this looks like a strong candidate for "ARIMA offers little
  advantage over a historical-mean or naïve baseline" — worth watching
  closely once the actual model-fitting/comparison stage runs, since a
  domain with near-zero autocorrelation is exactly where a mean baseline
  is hardest to beat.
