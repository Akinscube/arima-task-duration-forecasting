# Baseline Model Parameters (T1)

Companion to [baseline-forecasting-models.md](baseline-forecasting-models.md)
— the concrete constants and library calls `src/baselines.py` is built
against, decided here so code and doc can't drift apart silently.

## Train/test split

Reuses `data_quality.py`'s Gate 3 rule exactly:
`TRAIN_FRACTION = 0.8`, `n_train = int(n * TRAIN_FRACTION)`.

All four final datasets currently have `missing_pct = 0.0%` (confirmed by
both `data_quality.py`'s Gate 2 and every `reports/eda/<name>_eda.txt`), so
`n_train`/`n_test` counted this way are equivalent to taking the **first
`n_train` and last `n_test` positions of the full regular-calendar series**
— a genuine contiguous, time-ordered split (train = past, test = most
recent period), which is what walk-forward evaluation needs. This
equivalence depends on the 0%-missing precondition; if a future dataset
qualifies with nonzero missing periods, positional slicing would need
revisiting (flagged, not expected to matter for the current four).

## Moving-average window

`MA_WINDOW = 7`, fixed for every dataset regardless of `frequency` (see
`baseline-forecasting-models.md`'s design-decisions table for the
cross-domain-comparability rationale).

## Simple exponential smoothing

- Library call: `statsmodels.tsa.holtwinters.SimpleExpSmoothing(train,
  initialization_method="estimated").fit()` — the statsmodels-recommended
  initialization method (not the legacy heuristic), default optimizer.
- `α` = `fit_result.params["smoothing_level"]`, fit once on the training
  series only.
- Test-period walk-forward: the first test-point forecast is
  `fit_result.forecast(1)` (uses training data only, matching the
  no-lookahead constraint); each subsequent test-point forecast continues
  the recursion `ŷ_{t+1} = α·y_t + (1-α)·ŷ_t` manually, using the true
  revealed test actual `y_t` at each step — `α` itself is never
  re-estimated during this walk-forward continuation.

## MAPE zero-guard

Any test point with `|actual| <= 1e-9` is excluded from the MAPE
computation (its count is still reported in the per-dataset report, as
`mape_excluded_zero_actuals`). MAE and RMSE are unaffected — the guard
only applies to MAPE's division by the actual value.

## Output paths

- `reports/baselines/<name>_baselines.txt` — per-dataset report
- `reports/baselines/<name>_baselines_forecasts.csv` — per-test-point
  actual, forecast, and error for all four methods
- `reports/baselines/baselines_cross_domain_summary.txt` — all
  qualifying datasets x all four methods, MAE/RMSE/MAPE

## Error sign convention

`error = actual - forecast` (residual convention). MAE = mean(|error|),
RMSE = sqrt(mean(error²)), MAPE = mean(|error| / |actual|) x 100 over
non-zero actuals only.

## Series construction / quality gate

Reuses the same pattern already established by `src/eda.py`: import
`data_quality` directly and call its `load_config`/`profile` for the live
gate check (rule 1 — `data_quality.py` is the only quality screen, so the
gate determination itself must come from it, not be reimplemented);
duplicate the small load/resample/aggregate series-construction snippet
locally (matching `recover_structure.py` and `eda.py`'s existing
precedent), rather than importing `eda.py`'s version, to keep each stage
script decoupled from unrelated stage scripts.
