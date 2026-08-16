# ARIMA Fitting Parameters (T1)

Companion to [arima-fitting.md](arima-fitting.md) — the concrete constants
and library calls `src/arima_model.py` is built against, decided here so
code and doc can't drift apart silently.

## Train/test split

Reuses `data_quality.py`'s Gate 3 rule and `baseline-parameters.md`'s split
exactly: `TRAIN_FRACTION = 0.8`, `n_train = int(n * TRAIN_FRACTION)`,
positional first-`n_train`/last-`n_test` slicing (same 0%-missing
precondition already holds for all four final datasets).

## `d` selection (ADF, train-only)

- ADF significance level: **α = 0.05** (reused from `eda-parameters.md`,
  not re-decided) — but recomputed on the **training split only**, not
  read from `reports/eda/`, to keep the no-lookahead constraint airtight
  (see `arima-fitting.md`'s Design Decisions).
- `d = 0` if raw training-series ADF p ≤ α; else `d = 1` if
  first-differenced training-series ADF p ≤ α; else the stage fails
  explicitly for that dataset with `d>=2 (not resolved)` reported — this
  pipeline does not auto-difference past `d=1` (same boundary
  `cross-domain-eda.md` already set).

## `p`/`q` candidate bounding (ACF/PACF significance, train-only)

Computed on the training series after applying the `d` fixed above
(i.e. on the stationary/differenced training series, standard Box-Jenkins
identification practice).

- Significance band: **±1.96/√n_train** (the same 95% white-noise band
  `statsmodels.graphics.tsaplots.plot_acf`/`plot_pacf` use by default).
- Lag search range: `nlags = min(40, n_train // 2 - 1)` (identical formula
  to `eda-parameters.md`'s ACF/PACF `nlags`, applied here to the
  training-only series).
- `p_max` = the largest lag in `[1, nlags]` at which the **PACF** exceeds
  the significance band, capped at the global ceiling `P_MAX = 5`.
- `q_max` = the largest lag in `[1, nlags]` at which the **ACF** exceeds
  the significance band, capped at the global ceiling `Q_MAX = 5`.
- If no lag exceeds the band for either, the corresponding max is `0`
  (valid — collapses the grid toward `ARIMA(0,d,0)`, not an error).

## Grid search and order selection

- Candidate grid: every `(p, d_fixed, q)` for `p` in `0..p_max`, `q` in
  `0..q_max` (`d` is fixed, not searched — see `arima-fitting.md`
  Constraints).
- Library call: `statsmodels.tsa.arima.model.ARIMA(train, order=(p, d, q),
  trend=TREND, enforce_stationarity=True,
  enforce_invertibility=True).fit()`.
- `TREND` rule: `'c'` (constant) if `d == 0`, else `'n'` (no trend/constant)
  if `d >= 1` — no dataset in the final four has a planted deterministic
  linear trend beyond level/seasonal/AR components (`ground_truth.json`),
  so a drift term is not included even at `d=1`.
- A candidate is **excluded** from the AIC/BIC table (with the reason
  logged) if: the fit raises an exception, raises/triggers a
  `ConvergenceWarning`, or `enforce_stationarity`/`enforce_invertibility`
  reject it.
- **Selection criterion: AIC** (`fit_result.aic`), lowest wins among
  surviving candidates. BIC (`fit_result.bic`) is reported alongside for
  every surviving candidate; if the BIC-preferred order differs from the
  AIC-preferred order, both are stated in the report and the AIC-preferred
  order is what's used for fitting/forecasting (mirrors
  `recover_structure.py`'s existing "AIC-preferred order" precedent).
- If every candidate in the grid is excluded, the stage fails explicitly
  for that dataset (reported, not silently skipped).

## Final fit and walk-forward forecasting

1. Fit once at the selected order on the **full training set**:
   `model = ARIMA(train, order=selected_order, trend=TREND,
   enforce_stationarity=True, enforce_invertibility=True)`;
   `results = model.fit()`.
2. For each test-set point `t` in order:
   a. Forecast: `ŷ_t = results.forecast(1)` (uses only data already in
      `results`' state — training data plus any test points appended so
      far).
   b. Reveal the true `y_t`, then extend state:
      `results = results.append([y_t], refit=False)` — coefficients are
      **never** re-optimized; only the state is extended with the new
      observation, exactly mirroring the baseline stage's SES convention
      (`α` fit once, state updates only).
3. Order and coefficients from step 1 are frozen for the entire test
   window.

## MAPE zero-guard, error convention, output paths

Reused exactly from `baseline-parameters.md`:

- Any test point with `|actual| <= 1e-9` is excluded from MAPE (count
  reported as `mape_excluded_zero_actuals`); MAE/RMSE unaffected.
- `error = actual - forecast` (residual convention). MAE = mean(|error|),
  RMSE = sqrt(mean(error²)), MAPE = mean(|error| / |actual|) × 100 over
  non-zero actuals only.
- Output paths:
  - `reports/arima/<name>_arima.txt` — combined order-selection
    diagnostics + final metrics report
  - `reports/arima/<name>_arima_forecasts.csv` — per-test-point actual,
    forecast, error
  - `reports/arima/<name>_arima_forecast_plot.png` — actual-vs-forecast
    overlay over the test window
  - `reports/arima/arima_cross_domain_summary.txt` — all qualifying
    datasets, selected order, MAE/RMSE/MAPE (ARIMA only)
  - `reports/arima/arima_mape_comparison.png` — cross-domain MAPE bar
    chart (MAPE only — MAE/RMSE are in each dataset's native duration
    unit, not comparable across datasets; mirrors
    `baselines_mape_comparison.png`'s identical rationale)

## Series construction / quality gate

Same pattern as `eda.py`/`baselines.py`: import `data_quality` directly for
the live gate check (rule 1 — `data_quality.py` is the only quality
screen); duplicate the small load/resample/aggregate series-construction
snippet locally rather than importing another stage script's version,
keeping this stage decoupled per existing precedent.

## Global ceilings (tractability)

`P_MAX = 5`, `Q_MAX = 5` — caps the candidate grid at at most 6×6 = 36
fits per dataset. A dataset whose true significant-lag structure exceeds
this ceiling is a documented limitation to note in the report, not a
reason to raise the ceiling ad hoc mid-run.
