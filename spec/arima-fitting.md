# Spec: ARIMA Fitting and Tuning (Main Model)

## GOAL (restated)

Fit and tune a plain ARIMA(p,d,q) model per qualifying final dataset — order
chosen via ADF (for `d`), ACF/PACF (to bound candidate `p`/`q`), and AIC/BIC
(to select among candidates), exactly the diagnostic sequence `CLAUDE.md`
names — then produce walk-forward one-step-ahead test-set forecasts and
MAE/RMSE/MAPE, under the identical protocol already fixed by
`baseline-forecasting-models.md`, so the future RQ1 comparison and RQ2
cross-domain analysis are measuring the same thing on both sides.

## WHAT

For each of the four final datasets that currently passes the Stage-0
quality gate (same precondition `cross-domain-eda.md` and
`baseline-forecasting-models.md` established):

1. Reconstruct the series exactly as fixed by that dataset's
   `configs/<name>.yaml`, and split into train/test using the identical
   `TRAIN_FRACTION = 0.8` rule already validated by `data_quality.py`'s
   Gate 3 and reused by the baseline stage.
2. **Select the ARIMA order** using training data only:
   - Fix `d` via ADF (α = 0.05), recomputed on the training split (not
     read from `reports/eda/`, which was computed on the full series — see
     Design Decisions).
   - Bound candidate `p` and `q` via ACF/PACF significance on the
     training-differenced series (exact rule in `arima-parameters.md`).
   - Grid-search all `(p, d_fixed, q)` combinations in that bound, fit via
     `statsmodels.tsa.arima.model.ARIMA`, exclude non-converged or
     non-stationary/non-invertible fits, and select the order with the
     lowest AIC. Report BIC alongside; note if BIC prefers a different
     order.
3. **Fit once** at the selected order on the full training set. Order and
   coefficients are frozen after this single fit — never re-selected or
   re-optimized during the test phase (see Design Decisions:
   Walk-forward refit).
4. Produce **walk-forward, one-step-ahead** forecasts over the test window
   by extending the fitted model's state with each revealed test actual
   (`results.append([y_t], refit=False)`), forecasting one step ahead at
   each point — never refitting.
5. Compute MAE, RMSE, MAPE (same zero-guard convention as the baseline
   stage) and persist per-point forecasts/errors.
6. Write one per-dataset report (order-selection diagnostics + final
   metrics, combined — see Design Decisions), one forecast overlay plot,
   and one cross-domain summary (ARIMA's own results only — comparison
   against baselines is out of scope here), under `reports/arima/`.
7. **(Added after initial implementation, on request.)** Produce a
   cross-domain MAPE bar chart (`arima_mape_comparison.png`), mirroring
   `baselines.py`'s equivalent chart — MAPE, not MAE/RMSE, for the same
   reason: `build_series()` leaves durations in each dataset's native unit
   (hours/seconds/minutes), so MAE/RMSE are not comparable across rows.

## WHY

- `CLAUDE.md`'s pipeline names this exact sequence: "fit ARIMA (ADF,
  ACF/PACF, AIC/BIC)". This spec operationalizes that sequence precisely
  so the methodology chapter can point to a concrete, reproducible
  procedure rather than "ARIMA was fitted."
- **RQ1** needs ARIMA's MAE/RMSE/MAPE on the same walk-forward
  one-step-ahead protocol the four baselines already use — the baseline
  spec explicitly bound this stage to that protocol; breaking it would
  invalidate the comparison.
- **RQ2** asks how ARIMA's performance varies across domains with
  differing temporal properties. Two of the four datasets have
  EDA-confirmed seasonal structure (`generated_emergency_dept_stays`
  F_s=0.522, `call_center_emea_2021_2025` F_s=0.331) and two do not
  (`generated_job_shop_manufacturing` F_s=0.096,
  `synthetic_it_support_tickets` F_s=0.221). Using **plain ARIMA
  uniformly** (locked decision, not SARIMA on the seasonal two) means any
  underperformance on the seasonal datasets is a genuine, interpretable
  RQ2 finding about ARIMA's fit to that domain's temporal properties, not
  an artifact of giving different domains different model classes.

## DESIGN DECISIONS

| Decision | Options considered | Choice | Why |
|---|---|---|---|
| Seasonal component | SARIMA on EDA-confirmed-seasonal datasets vs. plain ARIMA uniformly vs. fit both | **Plain ARIMA(p,d,q) for all four, no seasonal term** | Matches `CLAUDE.md`'s proposal title and RQ1 wording exactly ("ARIMA", no seasonal qualifier). Keeps the model class identical across all four domains, which is what makes the RQ2 cross-domain comparison a comparison of *domain fit* rather than *model class*. Expected consequence: underperformance on the two seasonal datasets relative to baselines that partially track weekly pattern (MA, SES) — an anticipated, reportable RQ2 finding (see Risks), not something to fix by switching model class after seeing results. |
| Order-selection method | Automated search (`pmdarima.auto_arima`) vs. manual Box-Jenkins grid scored by AIC/BIC | **Manual grid: `d` fixed by ADF, `p`/`q` bounded by ACF/PACF significance, selected by AIC** | `CLAUDE.md` names the diagnostic sequence explicitly (ADF, ACF/PACF, AIC/BIC) — a black-box automated search would need separate justification that it satisfies the same requirement. Manual grid keeps every step traceable in the methodology chapter and avoids adding `pmdarima` as a dependency. |
| `d` selection | Reuse `reports/eda/<name>_eda.txt`'s already-computed `suggested_d` vs. recompute ADF on the training split only | **Recompute on training split only** | EDA's `suggested_d` was computed on the *full* series (train+test) — using it here would leak test-period information into a training-only decision, however small in practice. Recomputing on train-only data using the identical ADF α=0.05 rule (`eda-parameters.md`) keeps the no-lookahead constraint airtight. The two values may occasionally differ; if they do, both are reported, not silently reconciled. |
| `p`/`q` candidate bounds | Fixed global grid (e.g. always search 0..5) vs. ACF/PACF-significance-bounded per dataset | **ACF/PACF-significance-bounded per dataset, capped at a global ceiling** | A fixed blind 0..5×0..5 grid for every dataset ignores the diagnostic step `CLAUDE.md` explicitly names; bounding by significant lags on the training-differenced series makes the ACF/PACF step load-bearing, not decorative. The global ceiling (`P_MAX=Q_MAX=5`, see `arima-parameters.md`) keeps the grid tractable and reproducible even if a dataset shows a very long significant-lag tail. |
| AIC vs BIC as primary criterion | AIC only, BIC only, or report both with one as tie-break | **AIC primary, BIC reported alongside** | Matches `recover_structure.py`'s existing "AIC-preferred order" precedent (same criterion already used and reported in this pipeline). BIC is reported for transparency since it tends to prefer more parsimonious orders; if the two disagree, both are stated and AIC's pick is used, not hidden. |
| Walk-forward refit | Refit order+coefficients at every test step (expanding-window re-estimation) vs. fit once on train, extend state only | **Fit once on train; extend state only via `append(refit=False)`** | Mirrors the baseline stage's SES precedent exactly (`α` fit once, never re-estimated during walk-forward) — this is what keeps the RQ1 comparison apples-to-apples across all five methods. Per-step refitting would also multiply the order-selection grid search by `n_test`, which is neither what the baseline stage does for its one tunable parameter nor tractable to keep deterministic/reproducible. |
| Report structure | One combined per-dataset report (order-selection diagnostics + final metrics) vs. two separate files | **One combined report per dataset** | Mirrors `recover_structure.py`'s existing precedent (diagnostics and pass/fail in one report) more closely than the baseline stage's structure does, and keeps the order-selection rationale next to the metrics it produced, useful for the methodology write-up. |
| Cross-domain comparison scope | Include baselines in the ARIMA cross-domain summary vs. ARIMA-only | **ARIMA-only** (`reports/arima/arima_cross_domain_summary.txt` reports ARIMA's four datasets' metrics; no baseline columns) | Mirrors the baseline stage's own scope boundary exactly (its spec explicitly deferred "comparing baseline results against ARIMA" to a later stage). Combining them here would pre-empt the dedicated evaluation/significance-testing stage. |

## CONSTRAINTS

- Operates only on datasets that are both one of the four final datasets
  and currently passing `data_quality.py`'s gate.
- Uses each dataset's Stage-0 config exactly as already committed — no new
  frequency/aggregation/column decisions.
- Train/test split uses the identical `TRAIN_FRACTION = 0.8` rule already
  validated by `data_quality.py` and reused by the baseline stage.
- **No lookahead, strictly**: `d` selection (ADF), `p`/`q` bounding
  (ACF/PACF), grid search (AIC/BIC), and the initial coefficient fit all
  use training-split data only. Test data enters only as revealed
  one-step-ahead actuals during the walk-forward `append(refit=False)`
  phase.
- Plain ARIMA(p,d,q) only — no seasonal terms (locked decision above).
- `d` is fixed by ADF, not searched via AIC/BIC (matches `CLAUDE.md`'s
  stated order: ADF first, then ACF/PACF, then AIC/BIC over `p`/`q`).
- `enforce_stationarity=True` and `enforce_invertibility=True` for every
  candidate fit (statsmodels defaults, stated explicitly rather than
  silently relied upon) — non-stationary/non-invertible fits are excluded
  from the AIC/BIC comparison, not silently accepted.
- Order and coefficients are frozen after the single training fit; never
  re-selected or re-optimized during the test phase.
- Writes all output under `reports/arima/`, never into `reports/`,
  `reports/eda/`, or `reports/baselines/`.
- Does not modify `data_quality.py`, `recover_structure.py`, `eda.py`,
  `baselines.py`, `ground_truth.json`, or any existing config/spec —
  purely additive.
- No comparison against baseline results and no significance testing in
  this stage (Diebold–Mariano remains unconfirmed by supervisor per
  `CLAUDE.md`) — produces ARIMA results only, for the future evaluation
  stage to consume alongside the baseline stage's already-persisted
  per-point forecasts.

## RISKS

- **MLE optimizer non-determinism.** ARIMA fitting is a multi-parameter
  nonlinear optimization (more surface area than SES's single-parameter
  fit) — platform-dependent floating-point behavior is possible even with
  fixed inputs. Must be verified by re-running end-to-end and diffing
  output, not assumed reproducible by default (same class of risk the
  baseline spec flagged for SES, more pronounced here).
- **Plain ARIMA is expected to underperform on the two seasonal
  datasets** (`generated_emergency_dept_stays`,
  `call_center_emea_2021_2025`) relative to baselines that partially
  capture weekly pattern (moving average, SES). This is an anticipated,
  locked-in-advance consequence of the "plain ARIMA uniformly" decision —
  must be reported factually as an RQ2 finding, not investigated as a bug
  or used to justify switching to SARIMA after seeing results.
- **Grid candidates may fail to converge** (optimizer non-convergence,
  or violate stationarity/invertibility). Must be excluded from the
  AIC/BIC table with a logged reason, not crash the run or silently
  distort the comparison. If *every* candidate fails for a dataset, that
  dataset's ARIMA stage fails explicitly and is reported as such, not
  skipped silently.
- **`p_max = q_max = 0` edge case**: if no lag is significant beyond the
  ACF/PACF band on a dataset's training-differenced series, the grid
  collapses to the single candidate ARIMA(0,d,0). This is a valid outcome
  to report, not an error condition.
- **Train-only `d` may differ from EDA's full-series `suggested_d`** for
  a given dataset, since the two are computed on different spans (see
  Design Decisions). If this happens, both values are reported; the
  train-only value is what this stage uses, and the discrepancy itself is
  a data point worth noting in the write-up rather than silently
  resolved.
- **`append(refit=False)` assumes the fitted model stays adequate through
  the test window.** Not expected to break down given the recovery
  validation results already confirming stable AR/ARMA structure in the
  generated datasets, but if forecast error visibly drifts across the
  test window in the per-point CSV/plot, that should be reported as
  observed, not masked.

## SUCCESS / ACCEPTANCE CRITERIA

- ARIMA fitting runs only for datasets whose current `data_quality.py`
  verdict is `PASS`; a failing dataset is skipped and listed with its
  reason in the cross-domain summary (mirrors baseline-stage behavior).
- Per dataset, the combined report (`reports/arima/<name>_arima.txt`)
  states: train-only ADF p-values (raw and first-differenced) and the
  resulting `d`; the ACF/PACF significant-lag bounds and resulting
  `p_max`/`q_max`; the full AIC/BIC grid table with any excluded
  (non-converged) candidates and their exclusion reason; the selected
  order and whether AIC/BIC agreed; final MAE/RMSE/MAPE.
- `reports/arima/<name>_arima_forecasts.csv` persists per-test-point
  actual, forecast, and error (same convention as the baseline stage's
  CSVs).
- `reports/arima/<name>_arima_forecast_plot.png` overlays actual vs.
  ARIMA forecast over the test window.
- `reports/arima/arima_cross_domain_summary.txt` tabulates all qualifying
  datasets' selected order and MAE/RMSE/MAPE (ARIMA only, no baseline
  columns).
- Re-running end-to-end reproduces identical order selection and metrics
  (verified, not assumed, per the MLE-determinism risk above; any
  observed non-determinism is documented rather than hidden).
- No lookahead anywhere, verifiable by code inspection: every value used
  in `d`/`p`/`q` selection and the initial coefficient fit traces only to
  training-split rows.
- No exit code / pass-fail gate semantics of its own (consistent with the
  baseline and EDA stages — this stage produces numbers and a documented
  order choice, not a gate).

## TASKS

| # | Task | Depends on | Independently verifiable by |
|---|------|-----------|------------------------------|
| T1 | Write `arima-parameters.md`: `P_MAX`/`Q_MAX`, ACF/PACF significance-band formula and lag-search range, ADF α (reused), trend rule (`'c'` if `d=0` else `'n'`), `enforce_stationarity`/`enforce_invertibility` flags, AIC-primary/BIC-reported selection rule, exact `append(refit=False)` call sequence, output paths, MAPE zero-guard (reused). | — | Doc exists, covers all four datasets with no blanks, and every constant traces to a stated rationale. |
| T2 | Implement quality-gate check + series construction + train/test split helper (duplicate the `data_quality.py`-gate and series-construction pattern already established in `eda.py`/`baselines.py`, per the existing per-stage decoupling precedent). | — | Running it against each of the 4 configs reports today's PASS status and `n_train`/`n_test` sizes identical to the baseline stage's. |
| T3 | Implement train-only `d` selection via ADF (α=0.05), reproducing `eda-parameters.md`'s rule but on the training split only. | T2 | For each dataset, reported `d` and train-only ADF p-values are logged; a hand check on `generated_job_shop_manufacturing` (train-only) confirms `d=0` given its full-series ADF p≈0.0000 (expected to hold on the training subset too). |
| T4 | Implement ACF/PACF significant-lag bounding on the training-differenced series → `p_max`, `q_max` (capped at `P_MAX`/`Q_MAX`). | T3 | Given a hand-built AR(1)-with-known-phi synthetic series, the computed significant lags match the expected PACF cutoff at lag 1. |
| T5 | Implement the grid search: fit every `(p, d, q)` in bounds via `statsmodels.tsa.arima.model.ARIMA`, exclude non-converged/non-stationary/non-invertible fits with logged reasons, score by AIC (report BIC), select the order. | T4 | Against `generated_job_shop_manufacturing` (known-good ARMA(1,1) per `ground_truth.json` and already-passing recovery validation), the selected order is `(1,0,1)` or clearly justified if not. |
| T6 | Implement the fit-once-then-extend walk-forward forecaster: single training fit at the selected order, then `append([y_t], refit=False)` + one-step `.forecast(1)` per test point, no re-optimization. | T5 | On a small hand-checkable synthetic AR(1) series, the first few walk-forward forecasts match a manually computed AR(1) one-step prediction from the fitted coefficients. |
| T7 | Implement MAE/RMSE/MAPE computation and per-point forecast/error persistence, reusing the baseline stage's zero-guard and sign convention exactly. | T6 | Metrics on a hand-checkable synthetic series match manual calculation; a zero-actual test point triggers the guard rather than crashing or emitting `inf`. |
| T8 | Implement the combined per-dataset report (order-selection diagnostics + metrics), the forecast overlay plot, and the ARIMA-only cross-domain summary. | T7 | Reports exist under `reports/arima/` with all T1-listed content; cross-domain summary has one row per qualifying dataset. |
| T9 | Run end-to-end against all four final datasets; verify reproducibility by re-running and diffing output (addresses the MLE-determinism risk directly). | T2–T8 | Two consecutive runs produce identical (or explicitly documented near-identical, if non-deterministic) selected orders and metrics. |

**Ordering:** T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9.

## OUT OF SCOPE

- SARIMA or any seasonal ARIMA term (locked decision; revisiting this
  requires a new planning discussion, not a silent change after seeing
  results).
- `pmdarima.auto_arima` or any other automated/black-box order search.
- Comparing ARIMA results against the baseline stage's results, or any
  significance testing between methods — the next "evaluate model" stage
  (also still blocked on Diebold–Mariano confirmation per `CLAUDE.md`).
- Multi-step or long-horizon forecasting — only walk-forward one-step-ahead,
  per the baseline stage's binding protocol precedent.
- Re-deciding `d` via AIC/BIC — `d` comes from ADF only, per `CLAUDE.md`'s
  stated diagnostic order.
- Refitting coefficients or order during the test phase.
- Written interpretation of ARIMA's cross-domain performance pattern —
  this stage produces reproducible numbers and a documented order choice;
  cross-domain interpretation is the later "analyse and interpret
  results" stage.
