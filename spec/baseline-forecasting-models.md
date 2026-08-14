# Spec: Baseline Forecasting Models

## GOAL (restated)

Implement the four conventional baseline forecasters named in `CLAUDE.md`'s
RQ1 — naïve, historical mean, moving average, exponential smoothing — for
each qualifying final dataset, producing per-point test-set forecasts and
MAE/RMSE/MAPE summaries that the future ARIMA-fitting stage will be
compared against.

## WHAT

For each of the four final datasets that currently passes the Stage-0
quality gate (same precondition `cross-domain-eda.md` established):

1. Reconstruct the series exactly as already fixed by that dataset's
   `configs/<name>.yaml`, and split it into train/test using the same
   `TRAIN_FRACTION` rule `data_quality.py` already validated
   (`n_train = floor(n * 0.8)`).
2. Produce **walk-forward, one-step-ahead** forecasts over the test window
   for each of the four baselines (exact definitions below).
3. Compute MAE, RMSE, MAPE for each baseline, per dataset.
4. Persist **per-point forecasts and errors**, not just the summary
   metrics — the future evaluation/significance-testing stage needs paired
   errors and must not have to recompute baselines to get them.
5. Write a per-dataset report and one cross-domain summary, under a
   directory kept separate from `reports/`, `reports/eda/`, and the
   recovery-validation report (mirroring `cross-domain-eda.md`'s
   separation decision).
6. **(Added after initial implementation, on request.)** Produce two
   plots: a per-dataset actual-vs-forecast overlay over the test window
   (all four methods plus the actual series), and one cross-domain
   comparison chart. See Design Decisions for why the cross-domain chart
   uses MAPE rather than MAE/RMSE.

**Baseline definitions** (all walk-forward: at each test-set time `t`, a
method uses only `y_1 ... y_{t-1}` — train data plus any test points
already revealed — never future data):

| Method | Forecast at time t |
|---|---|
| Naïve | `ŷ_t = y_{t-1}` (last observed value) |
| Historical mean | `ŷ_t = mean(y_1, ..., y_{t-1})` (expanding window) |
| Moving average | `ŷ_t = mean(y_{t-k}, ..., y_{t-1})` (fixed window `k`, see Design Decisions) |
| Simple exponential smoothing | `ŷ_t = α·y_{t-1} + (1-α)·ŷ_{t-1}`, `α` fit once on the training set only, level state updates with each revealed actual during walk-forward |

## WHY

- **RQ1** asks whether ARIMA beats "naïve, historical mean, moving average,
  exponential smoothing" on MAE/RMSE/MAPE — these four methods, named in
  that exact wording, are the entire comparison target. Nothing about
  ARIMA can be evaluated against RQ1 until these exist.
- Walk-forward one-step-ahead evaluation is the standard, defensible
  protocol for this kind of comparison (Hyndman & Athanasopoulos) and
  matches operational reality for **RQ2**'s workforce-planning framing — a
  workforce planning tool would forecast one period ahead and update daily
  as real durations come in, not commit to one long-horizon forecast made
  once. This protocol choice **also binds the future ARIMA-fitting spec**:
  whatever protocol baselines use here is what ARIMA must use too, or the
  MAE/RMSE/MAPE comparison in RQ1 isn't measuring the same thing on both
  sides.
- Persisting per-point forecasts (not just summary metrics) is what makes
  the later significance-testing stage possible without re-deriving
  baseline forecasts from scratch.

## DESIGN DECISIONS

| Decision | Options considered | Choice | Why |
|---|---|---|---|
| Forecasting protocol | Fixed-origin static multi-step forecast (fit once, forecast the whole test horizon in one shot) vs. walk-forward one-step-ahead (re-forecast each test point using all data revealed so far) | **Walk-forward, one-step-ahead** | A static forecast over a long test horizon (e.g. ED's ~292-day test set) collapses 1-step and 292-step errors into one number, which tends to make every method converge toward "predict the unconditional mean" and washes out real differences between methods. Walk-forward keeps every forecast at a consistent, short, operationally realistic horizon. |
| Dataset selection | Run against the "final four" unconditionally vs. gate on the dataset's current quality-check verdict | **Gate on `data_quality.py`'s current `VERDICT: PASS`**, re-checked at run time | Same precondition established in `cross-domain-eda.md` for consistency — a dataset shouldn't enter modelling if it wouldn't have entered EDA. |
| Output location | `reports/` or `reports/eda/` vs. a dedicated directory | **`reports/baselines/`** | Mirrors `cross-domain-eda.md`'s separation decision — keeps each pipeline stage's artifacts from intermixing. |
| Moving-average window `k` | Fixed `k` uniform across all four datasets vs. `k = seasonal_period` per dataset (7 for daily, 52 for weekly) | **Fixed `k = 7` for every dataset, regardless of frequency** | A per-dataset `k = seasonal_period` would make `call_center_emea_2021_2025`'s moving average a 52-week (~1-year) smoother while the daily datasets get a 7-day smoother — structurally different baselines dressed up as "the same method," which would undermine the RQ2 cross-domain comparison. A fixed `k` is directly comparable across domains, at the acknowledged cost that a 7-week window means something different (a much shorter relative memory) for the weekly call-center series than a 7-day window does for the daily series — stated here as a limitation, not hidden. |
| Exponential smoothing parameter `α` | Fixed constant (e.g. 0.3) vs. fit per dataset by minimizing training-set SSE (standard SES fitting) | **Fit per dataset** via standard SES fitting on the training set only, held fixed through the walk-forward test phase (only the level state updates, not `α` itself) | A single fixed `α` across four structurally different domains (from `call_center_emea_2021_2025`'s near-noise weekly series to `generated_job_shop_manufacturing`'s high-persistence ARMA(1,1) series) would be arbitrary and likely poorly suited to at least some of them; fitting `α` per dataset is standard practice and keeps this baseline "simple exponential smoothing" rather than requiring the student to hand-pick four numbers. |
| Naïve variant | Plain (lag-1) naïve vs. seasonal naïve (`ŷ_t = y_{t-seasonal_period}`) | **Plain naïve** | `CLAUDE.md`'s RQ1 names "naïve" without qualifier, alongside three other specifically-named methods — substituting a seasonal variant would be answering a different, easier comparison than the one actually asked. Plain naïve is expected to perform worst on the two datasets EDA already found meaningful weekly seasonality in (`generated_emergency_dept_stays`, `call_center_emea_2021_2025`, per their `_eda_interpretation.md` files) — that's an honest, reportable RQ2-relevant finding, not a reason to swap the method. |
| Refitting during walk-forward | Refit every method (recompute mean/window/`α`) at every test step vs. no refitting, only state updates | **No refitting of fitted parameters** (`α` stays fixed post-training-fit); naïve/mean/moving-average are recomputed fresh from revealed data at each step because that's inherent to their definition, not a "refit" in the tuning sense | Keeps all four baselines under one consistent walk-forward contract, and keeps SES's `α` comparable to how it would be treated in the eventual ARIMA order-selection stage (fit once on train, not continuously re-optimized). |
| Cross-domain comparison metric (plot) | MAE/RMSE bar chart vs. MAPE bar chart | **MAPE** | `build_series()`'s duration is each dataset's *native* unit (hours for `synthetic_it_support_tickets`, seconds for `call_center_emea_2021_2025`, minutes for the two generated datasets) — this was never normalized the way `eda.py` converts to minutes for display, a gap not caught when this spec was first written. A bar chart of MAE/RMSE across datasets in three different units would visually mislead. MAPE is already a relative, unit-independent measure, so it sidesteps the problem without needing a unit-conversion pass — flagged here as a pre-existing gap rather than fixed as part of adding these plots; revisit if a unit-normalized MAE/RMSE cross-domain chart is wanted later. |

## CONSTRAINTS

- Operates only on datasets that are both one of the four final datasets
  and currently passing `data_quality.py`'s gate.
- Uses each dataset's Stage-0 config exactly as already committed — no new
  frequency/aggregation/column decisions.
- Train/test split uses the identical `TRAIN_FRACTION = 0.8` rule already
  validated by `data_quality.py`'s Gate 3 — not re-decided here.
- No lookahead: every forecast at time `t` uses only data through `t-1`.
- Writes all output under `reports/baselines/`, never into `reports/`,
  `reports/eda/`, or `reports/eda/eda_cross_domain_summary.txt`'s
  directory.
- Does not modify `data_quality.py`, `recover_structure.py`, `eda.py`,
  `ground_truth.json`, or any existing config/spec — purely additive.
- No ARIMA fitting or ARIMA-vs-baseline comparison happens in this stage —
  it produces baseline results only, for the future evaluation stage to
  consume.
- Deterministic: SES's `α` fit must use a fixed, documented optimizer
  configuration so re-running produces identical output (flagged as a risk
  below — needs verifying, not assuming).

## RISKS

- **SES fitting may not be perfectly deterministic** depending on the
  optimizer statsmodels' `SimpleExpSmoothing` uses internally (numerical
  optimization can have platform-dependent floating-point behavior even
  with fixed inputs). Must be verified by re-running and diffing output,
  not assumed reproducible by default.
- **MAPE is undefined/unstable near-zero actuals.** All four datasets'
  EDA-reported duration distributions are comfortably positive (minutes,
  no near-zero medians), so this is not expected to bite — but the
  implementation must still guard it explicitly (skip/flag any test point
  with `y_t` at or near zero rather than silently producing `inf` or
  crashing).
- **Fixed `k=7` moving-average window is a documented cross-domain
  asymmetry** (see Design Decisions) — must be stated plainly in the
  cross-domain summary, not just in this spec, so it isn't lost by the
  time RQ2 analysis happens.
- **Plain naïve is expected to underperform** on datasets with confirmed
  weekly seasonality — this is an anticipated, not surprising, result;
  must be reported as such rather than treated as something to
  investigate or fix.
- **This stage's protocol choice is a binding precedent.** If the future
  ARIMA-fitting spec chooses a different forecasting protocol (e.g.
  static multi-step) without revisiting this decision, the RQ1 comparison
  becomes invalid. Flagging here so it's checked explicitly when that spec
  is written, not assumed compatible.

## SUCCESS / ACCEPTANCE CRITERIA

- Baselines run only for datasets whose current `data_quality.py` verdict
  is `PASS`; a failing dataset is skipped and listed with its reason in
  the cross-domain summary.
- For each qualifying dataset: `reports/baselines/<name>_baselines.txt`
  states MAE/RMSE/MAPE for all four methods, and
  `reports/baselines/<name>_baselines_forecasts.csv` (or equivalent)
  persists per-test-point actual value, and each method's forecast and
  error.
- `reports/baselines/baselines_cross_domain_summary.txt` tabulates all
  four datasets × four methods on the same metrics.
- Re-running against the committed CSVs reproduces identical MAE/RMSE/MAPE
  (verified, not assumed, per the SES determinism risk above).
- No exit code / pass-fail semantics of its own (consistent with EDA;
  this stage produces numbers, not gates).
- Every design decision above (window size, `α`-fitting method, protocol)
  is stated in the per-dataset report, not only implied by the numbers.

## TASKS

| # | Task | Depends on | Independently verifiable by |
|---|------|-----------|------------------------------|
| T1 | Write a short parameters reference (mirroring `eda-parameters.md`): MA window `k`, SES fitting method/library call, train/test split rule (reused, not new), output paths, MAPE zero-guard rule. | — | Doc exists, covers all four datasets and all four methods with no blanks. |
| T2 | Implement the quality-gate check + series construction + train/test split helper (reusing the `data_quality.py`-gate and series-construction precedent from `eda.py`). | — | Running it against each of the 4 configs reports today's PASS status and correct `n_train`/`n_test` split sizes matching `data_quality.py`'s own Gate 3 numbers. |
| T3 | Implement the four baseline forecast functions under the walk-forward, one-step-ahead protocol from T1. | T1, T2 | Given a small hand-checkable synthetic series, each method's forecasts match a manually computed expectation for the first few test points. |
| T4 | Implement MAE/RMSE/MAPE computation plus per-point forecast/error persistence, including the MAPE zero-guard. | T3 | Metrics computed against a hand-checkable synthetic series match manual calculation; a synthetic series with a zero actual triggers the guard rather than crashing or emitting `inf`. |
| T5 | Implement per-dataset report + cross-domain summary generation. | T4 | Reports exist under `reports/baselines/` with all T1-listed content; cross-domain summary has one row per qualifying dataset per method. |
| T6 | Run end-to-end against all four final datasets; verify reproducibility by re-running and diffing output (addresses the SES-determinism risk directly rather than assuming it). | T2–T5 | Two consecutive runs produce byte-identical (or explicitly documented near-identical, if SES proves non-deterministic) output. |

**Ordering:** T1 → T2 → T3 → T4 → T5 → T6.

## OUT OF SCOPE

- ARIMA order selection, fitting, or forecasting (the next stage, "train
  and tune main model").
- Comparing baseline results against ARIMA, or any significance testing
  between methods (the stage after that, "evaluate model" — also still
  blocked on Diebold–Mariano confirmation per `CLAUDE.md`).
- Written interpretation of which baseline "wins" per domain — this stage
  produces reproducible numbers; cross-domain interpretation is the later
  "analyse and interpret results" stage.
- Holt/Holt-Winters or any trend/seasonal exponential-smoothing variant —
  `CLAUDE.md` names plain "exponential smoothing"; adding trend/seasonal
  components would blur the line between "simple baseline" and "the kind
  of structure ARIMA is supposed to be tested against."
