# EDA Interpretation — call_center_emea_2021_2025

Companion to [call_center_emea_2021_2025_eda.txt](call_center_emea_2021_2025_eda.txt)
and its five plots. No ground truth exists for this dataset (it's the
real, EMEA-filtered call-centre extract described in
[spec/call_center_data_prep.md](../../spec/call_center_data_prep.md)) — every
statement below is a description of what the data shows, not a validated
"recovery" of anything.

**Important caveat carried over from data prep:** `aHT_Seconds` is already
a *bucket-level average* handle time (one row = one `Date, Hour, Queue,
Site, Channel...` combination), not an individual call's duration. So even
the "record-level" statistics below describe a distribution of averages,
not of raw call durations — expect it to look smoother/less extreme than
a true per-call distribution would.

## Descriptive statistics

- **Record-level** (n=5,366 buckets): mean 9.07 min, sd 3.46, CV 0.38,
  median 9.1 min, p99 14.9 min, skew ≈ 0 (−0.03).
- **Series-level** (n=262 weeks, weekly median): mean 9.11 min, sd 1.21,
  CV 0.13, skew 0.07.

The CV drops from 0.38 to 0.13 going from record to series level — taking
a weekly median across ~20 buckets/week cuts the spread to roughly a
third, as expected from aggregation. The near-zero skew at both levels
means this is the most symmetric of the four datasets — no long tail to
worry about.

## Histogram (`_hist.png`)

Roughly flat/broad across 3–15 minutes rather than a single sharp peak.
Combined with the bucket-averaging caveat above, this reads as "many
different queue/site/channel combinations, each with its own typical
handle time, pooled together" — the histogram is a mixture of several
underlying distributions, not one homogeneous process. This matches
`spec/call_center_data_prep.md`'s own documented tradeoff: this series is
"aggregate EMEA support-operations handle time... not a single queue's
behavior."

## Time series + rolling mean (`_series.png`)

Highly volatile week-to-week (roughly 6–12.5 min), with a rolling mean
that stays almost flat around 9 minutes for the full 2021–2025 span — no
visible long-run trend or level shift. This is the noisiest-relative-to-
its-own-mean series of the four in raw appearance, even though its CV
(0.13) is actually the second-lowest — the visual noise is real week-to-
week variance, not a scale artifact.

## Seasonal boxplot by month (`_seasonal_boxplot.png`)

All twelve months show heavily overlapping boxes (medians cluster tightly
between 8.7 and 9.5 minutes) — no month stands out. This is a visually
**weak** seasonal signal, which sits in tension with the STL classification
below; see the note there.

## ACF / PACF (`_acf_pacf.png`)

Raw series: essentially every lag sits inside or right at the edge of the
95% confidence band, with one modest exception at **lag 7** (weeks) —
ACF 0.137, PACF 0.149. That's a mild "every ~7 weeks" echo, not a strong
one. First-differenced (d=1): a sharp negative spike at lag 1 (≈−0.53) in
both ACF and PACF, then a decaying string of small negative lags in
PACF(d=1) — the textbook signature of **over-differencing**. The raw
series was already stationary (ADF d=0 p≈0.0000), so differencing it adds
structure rather than removing it. `suggested_d = 0` is the right call
here.

## STL decomposition (`_stl.png`)

Trend: a slow multi-year wave, roughly 8.9–9.3 min, bottoming out in late
2022 and peaking in 2024. Season: small ripples, roughly ±2–3 min.
Residual: the dominant component by far, ±3–4 min, sometimes larger.
`seasonal_strength (F_s) = 0.331`, just over the 0.3 threshold, so it's
tagged `visible_seasonality: y` — but given how weak and noisy the season
panel looks next to the residual panel, and how flat the monthly boxplot
is, **treat this as a borderline "yes," not a strong one.** The
threshold-based label and the visual impression disagree somewhat; both
facts are worth reporting rather than picking one.

## What this suggests going into ARIMA fitting

- Low apparent temporal structure: the strongest thing in the raw ACF/PACF
  is a modest weekly (lag-7) echo, everything else is noise-band.
- `d=0` is well-supported (ADF already rejects a unit root; differencing
  actively hurts).
- The weakest, most ambiguous seasonality signal of the four datasets —
  worth flagging for the RQ2 cross-domain discussion as a domain where
  ARIMA has the least seasonal structure to exploit, and where a simple
  baseline (e.g. historical mean) may be competitive precisely because the
  series is close to noise around a stable weekly-ish level.
