# ARIMA (p,d,q) Parameter Selection Rationale

Consolidated methodology-and-results documentation for the ARIMA
order-selection step of the model-architecture/fitting stage. Methodology
is fixed in [spec/arima-fitting.md](../../spec/arima-fitting.md) and
[spec/arima-parameters.md](../../spec/arima-parameters.md); the numbers
below are reproduced from each dataset's generated report
(`reports/arima/<name>_arima.txt`), written fresh by `src/arima_model.py`
on every run — this document is a readable summary of those reports, not
an independent source of truth.

## Method (applies identically to all four datasets)

1. **`d` (differencing order) — via ADF, train-only.** Augmented
   Dickey-Fuller test at α=0.05, run on the training split only (not the
   full series — see the "train-only `d`" design decision in
   `arima-fitting.md` for why this can occasionally diverge from EDA's
   full-series `suggested_d`). `d=0` if the raw training series is
   stationary; else `d=1` if the once-differenced training series is
   stationary; `d≥2` is not attempted (would fail the stage explicitly —
   did not occur for any of the four datasets).
2. **`p`/`q` bounds — via ACF/PACF significance, train-only.** Computed on
   the training series after applying the `d` fixed in step 1. A lag is
   "significant" if its ACF/PACF value exceeds ±1.96/√n_train (the
   standard 95% white-noise band). The largest significant PACF lag caps
   candidate `p`; the largest significant ACF lag caps candidate `q`; both
   are additionally capped at the global ceiling `P_MAX=Q_MAX=5` for
   tractability (`spec/arima-parameters.md`).
3. **Order selection — via AIC/BIC grid search.** Every `(p, d_fixed, q)`
   combination in the resulting bounds is fit
   (`statsmodels.tsa.arima.model.ARIMA`, `enforce_stationarity=True`,
   `enforce_invertibility=True`); non-converged fits are excluded and
   logged. The surviving candidate with the lowest **AIC** is selected;
   **BIC** is reported for every surviving candidate, and a note is added
   if BIC would have preferred a different (typically more parsimonious)
   order.

No seasonal term is used anywhere (locked decision — plain ARIMA for all
four datasets, see `arima-fitting.md`'s Design Decisions), so any weekly
pattern a dataset has must be approximated, if at all, by higher-order
AR/MA terms rather than a seasonal component.

## Per-dataset results

### `generated_job_shop_manufacturing`

| Step | Result |
|---|---|
| `d` | ADF d=0 p=0.0000, d=1 p=0.0000 → **d=0** (already stationary) |
| ACF/PACF bounds | band=±0.0574. PACF significant: [1,2,3,4,26] → p_max=5. ACF significant: [1,2,...,18,30,...,40] → q_max=5 |
| Selected order | **(1,0,1)**, AIC=4971.74, BIC=4991.99 — AIC and BIC agree |
| Final metrics | MAE=1.6362, RMSE=2.0706, MAPE=2.96% |

**Note:** (1,0,1) exactly matches this dataset's *planted* ground-truth
ARMA(1,1) structure (`ground_truth.json`), and agrees with
`recover_structure.py`'s independently-computed recovery-validation
AIC-preferred order. This is the strongest cross-check in the whole
pipeline that the ACF/PACF/AIC selection procedure works as intended.

### `generated_emergency_dept_stays`

| Step | Result |
|---|---|
| `d` | ADF d=0 p=0.0970 (non-stationary), d=1 p=0.0000 (stationary) → **d=1** |
| ACF/PACF bounds | band=±0.0574. PACF significant: 20 lags in [1,39] → p_max=5 (capped). ACF significant: 21 lags in [1,40] → q_max=5 (capped) |
| Selected order | **(5,1,1)**, AIC=9855.88, BIC=9891.31 — AIC and BIC agree |
| Final metrics | MAE=13.2721, RMSE=16.8442, MAPE=7.66% |

**Note:** this is the one dataset with EDA-confirmed strong seasonality
(F_s=0.522). The ACF/PACF significant-lag lists are dense and span far
beyond `P_MAX`, reflecting weekly structure that a non-seasonal ARIMA can
only approximate with high-order AR terms — order search saturated at the
`p_max=5` ceiling as a direct symptom of this.

### `call_center_emea_2021_2025`

| Step | Result |
|---|---|
| `d` | ADF d=0 p=0.0000, d=1 p=0.0000 → **d=0** |
| ACF/PACF bounds | band=±0.1356 (wider — n_train=209 is much smaller, weekly series). PACF significant: [7,35,36] → p_max=5. ACF significant: [7] → q_max=5 |
| Selected order | **(0,0,0)**, AIC=2394.85, BIC=2401.53 — AIC and BIC agree |
| Final metrics | MAE=55.8782, RMSE=68.1435, MAPE=10.72% |

**Note:** the grid search's own AIC found no candidate with `p>0` or
`q>0` that improved on the trivial constant-mean model — consistent with
this dataset's EDA description as a near-noise weekly series (borderline
seasonality, F_s=0.331). ARIMA effectively degenerates to "predict the
training mean" here.

### `synthetic_it_support_tickets`

| Step | Result |
|---|---|
| `d` | ADF d=0 p=0.0000, d=1 p=0.0000 → **d=0** |
| ACF/PACF bounds | band=±0.0574. PACF significant: [2,6,11,14,15,30,35] → p_max=5. ACF significant: [2,6,11,14,25,30,35] → q_max=5 |
| Selected order | **(5,0,2)**, AIC=7473.16, BIC=7518.73 — **AIC and BIC disagree**: BIC prefers (0,0,0) |
| Final metrics | MAE=4.8284, RMSE=5.9311, MAPE=16.68% |

**Note:** this is the only dataset where AIC and BIC land on different
orders. Per the locked selection rule, AIC's pick (5,0,2) is what's used
for fitting/forecasting, but BIC's preference for the trivial (0,0,0)
model — a heavier penalty for the 7 extra parameters in (5,0,2) — suggests
the additional AR/MA terms buy only a marginal likelihood improvement.
Combined with this dataset's weakest MAPE (16.68%) of the four, this
reads as ARIMA finding comparatively little real structure to exploit
here.

## Cross-dataset pattern

| Dataset | `d` | Selected order | AIC/BIC agree | MAPE |
|---|---|---|---|---|
| `generated_job_shop_manufacturing` | 0 | (1,0,1) | yes | 2.96% |
| `generated_emergency_dept_stays` | 1 | (5,1,1) | yes | 7.66% |
| `call_center_emea_2021_2025` | 0 | (0,0,0) | yes | 10.72% |
| `synthetic_it_support_tickets` | 0 | (5,0,2) | **no** | 16.68% |

The dataset with a genuinely planted ARMA structure and no seasonality
(job-shop) is where order selection is cleanest and most confident (AIC/BIC
agree, order matches ground truth exactly). The two datasets with weak or
absent real structure (call-center's near-noise series, IT-support's
disagreement between AIC and BIC) are where order selection is least
decisive — an early, order-selection-level signal of the same RQ2 pattern
later visible in the MAPE comparison
(`reports/arima/arima_cross_domain_summary_interpretation.md`).
