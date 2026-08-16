# Model Evaluation Parameters (T1)

Companion to [model-evaluation.md](model-evaluation.md) — the concrete
constants and library calls `src/evaluate_model.py` is built against,
decided here so code and doc can't drift apart silently.

## Data sources (read-only)

- `reports/arima/<name>_arima_forecasts.csv` — columns `actual`,
  `arima_forecast`, `arima_error`.
- `reports/baselines/<name>_baselines_forecasts.csv` — columns `actual`,
  `<method>_forecast`, `<method>_error` for `method` in `naive`,
  `historical_mean`, `moving_average`, `ses`.
- A dataset is evaluated only if both files exist for it (i.e. it
  succeeded, non-skipped/non-failed, in both prior stages). Otherwise
  skipped with a stated reason — no CSV is generated or re-derived here.

## Index-alignment precondition

Before computing anything, the two CSVs' index (test-window dates) must
be identical: same length, same values, same order. If not, the dataset
is reported as a **failure** for this stage (`reason: "test-window index
mismatch between arima and baselines CSVs"`), not reindexed, not
truncated to the intersection, not silently joined on position.

## Loss differential, autocorrelation diagnostic, and significance test

**Status: Diebold–Mariano, provisional pending supervisor confirmation**
(supersedes the original Wilcoxon plan — see `model-evaluation.md`'s
Status note). Every report this stage writes must state this plainly.

- Per baseline, per test point: `d_t = |baseline_error_t| - |arima_error_t|`
  (paired absolute-error differential — positive means ARIMA's absolute
  error was smaller at that point).
- **Autocorrelation diagnostic** (reported for every pair, not only where
  it's significant): `statsmodels.stats.diagnostic.acorr_ljungbox(d_t,
  lags=[1, 5, 7], return_df=True)`. Report the lag-1, lag-5, lag-7
  p-values; flag the pair as "autocorrelated" if any is `< ALPHA`.
- **DM test**, implemented as an OLS regression of `d_t` on a constant
  with HAC (Newey–West) standard errors — mathematically the standard DM
  statistic (mean loss differential over its own robust standard error):
  ```python
  import statsmodels.api as sm
  X = np.ones(len(d_t))
  L = int(np.floor(4 * (len(d_t) / 100) ** (2 / 9)))  # Newey-West automatic bandwidth
  fit = sm.OLS(d_t, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
  dm_stat, p_value = float(fit.tvalues[0]), float(fit.pvalues[0])
  ```
  `L` is recomputed per (dataset, baseline) pair from that pair's own
  `n_test` — not a single global constant, since `call_center_emea_2021_2025`
  (n_test=53) and the three daily datasets (n_test≈292) warrant different
  bandwidths under the same rule.
- If every `d_t` is exactly zero for a pair (all forecasts identical), the
  OLS/HAC fit is degenerate — report that pair's row as
  `"identical forecasts, no test possible"` rather than letting the call
  produce a spurious result.

## Multiple-comparisons correction

- `statsmodels.stats.multitest.multipletests(pvals, alpha=0.05,
  method="holm")`, applied **once per dataset** to that dataset's 4 raw
  p-values (naïve, historical mean, moving average, SES) — never pooled
  across datasets.

## Significance threshold

`ALPHA = 0.05` for both the raw and Holm-adjusted p-value (matches the ADF
α already used throughout `eda.py`/`arima_model.py`).

## Win / loss / tie classification

For each (dataset, baseline) pair:

- **`arima_wins`**: ARIMA's MAE < baseline's MAE, AND Holm-adjusted p <
  `ALPHA`.
- **`baseline_wins`**: baseline's MAE < ARIMA's MAE, AND Holm-adjusted p <
  `ALPHA`.
- **`no_significant_difference`**: Holm-adjusted p >= `ALPHA` (regardless
  of which point estimate is lower).

MAE (not RMSE/MAPE) is the point-estimate metric used for this
classification, matching the absolute-error-based loss differential the
DM test itself uses (see `model-evaluation.md`'s design decision on
loss-function scope).

## Output paths

- `reports/evaluation/<name>_evaluation.txt` — per-dataset MAE/RMSE/MAPE
  table (ARIMA + 4 baselines), Ljung-Box diagnostic, raw/Holm-adjusted DM
  p-values and verdict per baseline pair, "provisional pending supervisor
  confirmation" label.
- `reports/evaluation/evaluation_cross_domain_summary.txt` — one row per
  evaluated dataset: counts of `arima_wins` / `baseline_wins` /
  `no_significant_difference` across its 4 pairs.
- `reports/evaluation/evaluation_win_loss_heatmap.png` — dataset ×
  baseline grid, cell color = verdict (win/tie/loss), annotated with the
  Holm-adjusted p-value.

## Series construction / quality gate

Not applicable — this stage does not touch `data_quality.py` or any raw
series. It reads only the two prior stages' already-persisted per-point
CSVs (see Data sources above), consistent with those stages' own stated
rationale for persisting them.
