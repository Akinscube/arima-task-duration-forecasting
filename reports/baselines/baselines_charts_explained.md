# Baseline Charts Explained

Companion to the five plots in `reports/baselines/`: four per-dataset
actual-vs-forecast overlays and one cross-domain MAPE comparison. Numbers
referenced below are from each dataset's `<name>_baselines.txt`. This
explains what the charts show; it does not interpret them against RQ1/RQ2
— that's the later "analyse and interpret results" stage, which needs
ARIMA's results alongside these to mean anything.

## How to read every actual-vs-forecast plot

Black = actual test-period series. The four colored lines are each
method's one-step-ahead forecast at every point (walk-forward: each
forecast only uses data up to the point before it — see
`spec/baseline-forecasting-models.md`). A method whose line hugs black
closely is tracking well; a flat line means that method isn't reacting to
recent movement at all.

## `synthetic_it_support_tickets_forecasts_plot.png`

Actual (black) is extremely spiky, jumping ±10 hours day to day with no
visible pattern — consistent with this dataset's near-zero ACF/PACF found
during EDA. **Naive (blue) sits almost exactly on top of actual**,
shifted by one day — because the series has no real structure to exploit,
"yesterday's value" ends up being a surprisingly close visual match purely
by chance/noise, even though its MAPE (23.2%) is still the worst of the
four methods numerically. **Historical mean (orange) and SES (red) are
both flat lines at ~30 hours** — SES's fitted α is ≈0 here, so it
collapses to essentially the same thing as the historical mean. **Moving
average (green)** is the only method with a visible, gentle wave — it
reacts to recent short-term drift but far more slowly than naive.

## `call_center_emea_2021_2025_forecasts_plot.png`

Similar story to the IT-support dataset: actual (black) is jagged. Naive
(blue) also zigzags — but because the true series is noisy rather than
patterned, tracking it one week late means naive is often trending the
*wrong direction* at any given point (visibly criss-crossing black rather
than hugging it), which is why naive is clearly the worst line to look at
here, matching its highest MAPE (14.1%) on the cross-domain chart.
Historical mean and SES again collapse to nearly the same flat line
(~550 seconds) — SES's α is ≈0 for this dataset too.

## `generated_emergency_dept_stays_forecasts_plot.png`

Actual (black) shows a clear repeating weekly sawtooth on top of a slow
multi-month wave — this is the planted weekend effect and annual cycle
EDA already identified. **Naive (blue) tracks the sawtooth closely**,
because a strong day-to-day pattern means "yesterday" is often a decent
predictor of "today" even without modelling the pattern explicitly.
**Historical mean (orange) is flat around 178 minutes** and visibly misses
both the weekly sawtooth and the slow seasonal drift entirely — it has no
way to react to either. **Moving average (green) and SES (red) both
smooth through the sawtooth**, producing a wavy line that follows the
slower seasonal trend reasonably well while averaging out the day-to-day
weekend spikes — visually the best compromise of the four here, matching
their lower MAE/MAPE numbers (SES: MAPE 8.3%, the best of the four).

## `generated_job_shop_manufacturing_forecasts_plot.png`

**Naive (blue) and SES (red) are visually identical** — SES's line is
drawn on top of naive's and completely hides it, because this dataset's
fitted α is exactly 1.0 (SES mathematically degenerates to naive when
α=1). Both track the actual series' multi-week swings closely, reflecting
the strong underlying persistence (ACF lag-1 = 0.90, per this dataset's
EDA). **Historical mean (orange) is flat at ~55 minutes** and clearly the
worst visual fit — it completely misses every swing, consistent with it
having by far the highest MAPE (6.4%) of the four methods on this
dataset. **Moving average (green)** partially tracks the swings but lags
and under/overshoots the peaks and troughs.

## `baselines_mape_comparison.png`

Grouped bars, one group per dataset, MAPE (%) on the y-axis (the one
metric safe to compare across datasets without unit conversion — see
`spec/baseline-forecasting-models.md`'s design-decisions table for why
MAE/RMSE aren't used here). The pattern that stands out: **naive is the
tallest (worst) bar in three of four datasets** —
`synthetic_it_support_tickets`, `call_center_emea_2021_2025`, and
`generated_emergency_dept_stays` — and **tied for the shortest (best) bar**
in the fourth, `generated_job_shop_manufacturing`. That flip lines up
exactly with each dataset's underlying temporal structure: naive does
badly on noisy/seasonal series where "yesterday's value" isn't a good
predictor of "today," and does well on `generated_job_shop_manufacturing`
specifically because that series is dominated by strong short-range
persistence (high AR(1)-like autocorrelation) — the one case where "just
use yesterday's value" is close to the mathematically optimal one-step
forecast.
