# Spec: Cross-Domain Exploratory Data Analysis (EDA)

## WHAT

Build a new, additive diagnostic stage that runs against **only the
datasets that have passed the Stage-0 quality gate** — i.e. every dataset
among the four final candidates declared in `CLAUDE.md` —

- `synthetic_it_support_tickets`
- `call_center_emea_2021_2025`
- `generated_emergency_dept_stays`
- `generated_job_shop_manufacturing`

— whose most recent `src/data_quality.py` run against its
`configs/<name>.yaml` reports `VERDICT: PASS` (i.e. `reports/<name>_quality.txt`
exists and its exit code was 0). Re-verified on 2026-08-12: **all four
currently pass.** If a dataset's config or source data later changes such
that it fails the gate, EDA must not run against it until it passes again
— the quality gate is a hard precondition, not merely informative context.

For each qualifying dataset, using the series exactly as already
constructed by its `configs/<name>.yaml` (same `timestamp_column`,
`duration_column`, `frequency`, `aggregation`, `seasonal_period` already
fixed at Stage 0), produce:

**Basic EDA:**
1. Descriptive statistics of the duration series (mean, sd, CV, quantiles,
   skewness) — both at the record level (pre-aggregation) and the
   constructed-series level (post-aggregation).
2. A missingness/outlier recap (reused from the dataset's existing
   `reports/<name>_quality.txt`, not recomputed independently).
3. A distribution plot (histogram) of record-level durations.
4. The time series plot with rolling mean (same construction as
   `data_quality.py`'s Step 4 plot, redrawn here so it lives alongside
   the rest of the EDA output rather than only in the quality report).
5. A seasonal subseries boxplot (by day-of-week for daily-frequency
   datasets, by month for weekly-frequency datasets).

**Diagnostic EDA (ARIMA-identification-facing):**
6. An ADF stationarity test on the raw series (d=0) and on the
   first-differenced series (d=1).
7. ACF and PACF plots (raw and first-differenced).
8. A seasonal-trend decomposition (STL) using the dataset's configured
   `seasonal_period`.

**Cross-domain:**
9. A per-dataset text report and plot set, written to a directory kept
   **separate** from Stage-0 quality reports and the recovery-validation
   report (see Output location below).
10. One **cross-domain summary table** placing all qualifying datasets
    side by side on the same metrics — the evidence base for RQ2's
    "differing temporal properties" comparison.

This is a diagnostics stage, not a gate: it produces no pass/fail verdict
and no exit-code gating of its own (it only *consumes* the Stage-0 gate's
result as a precondition), and it does not select or fit an ARIMA order
for the eventual baseline comparison — that is a separate, later pipeline
stage.

## WHY

- **RQ1** needs correctly identified ARIMA orders; ADF/ACF/PACF/decomposition
  are the standard Box-Jenkins identification diagnostics that must exist
  *before* model fitting is defensible.
- **RQ2** asks how ARIMA performance varies across domains with "differing
  temporal properties," and which domain it underperforms in. That claim
  needs a systematic, like-for-like comparison of temporal properties
  across all four domains — not an impression formed while fitting models
  one at a time. The cross-domain summary table is that evidence.
- Two of the four datasets (`generated_emergency_dept_stays`,
  `generated_job_shop_manufacturing`) already have a recovery-validation
  report because their DGP is known. The other two
  (`synthetic_it_support_tickets`, `call_center_emea_2021_2025`) have no
  ground truth — EDA is their *only* pre-modelling diagnostic step, so it
  needs to be done to the same standard as the synthetic pair, not as an
  afterthought.

## DESIGN DECISIONS

| Decision | Options considered | Choice | Why |
|---|---|---|---|
| New script vs. extend `data_quality.py` | Add ADF/ACF/decomposition to `data_quality.py` vs. a new `src/eda.py` | **New `src/eda.py`** | Rule 1 makes `data_quality.py` the *only quality screen* (a pass/fail gate). EDA has no gate semantics — folding it in would blur that boundary and risk two scripts implicitly gating the same data differently. |
| Dataset selection | Run EDA on the "final four" list unconditionally vs. gate on the dataset's current quality-check verdict | **Gate on `data_quality.py`'s current `VERDICT: PASS`** for each of the final four, checked at EDA run time (not hardcoded from a past run) | Explicit instruction: EDA should only ever look at data that has cleared the quality gate. Hardcoding "these four always qualify" would silently stop being true if a config or source file changed later; checking the verdict live keeps the precondition honest. |
| Output location | Write EDA reports/plots into `reports/` alongside `<name>_quality.txt`, `<name>_series.png`, and `recovery_validation.txt` vs. a dedicated subdirectory | **`reports/eda/`** — a separate subdirectory, mirroring the dataset-name convention inside it (`reports/eda/<name>_eda.txt`, `reports/eda/<name>_*.png`) | Explicit instruction: EDA output must be kept separate from the Stage-0 quality reports and the recovery-validation report, so the three diagnostic stages (screen / recover / explore) don't intermix their artifacts in one flat directory. |
| Series construction | Import `data_quality.py`'s load/resample helpers vs. duplicate the same 3-line resample logic locally | **Duplicate, with an explicit comment pointing at `data_quality.py`/`spec/quality-gate-thresholds.md` as the source of truth** | Matches the precedent already set by `recover_structure.py`'s `build_series()`, which does the same thing for the same reason (documented in its own docstring: "same construction as data_quality.py"). Keeps `eda.py` free of import side effects (matplotlib backend, argv handling) from an unrelated gating script. |
| Differencing depth | Adaptive (difference until ADF passes, up to some max d) vs. fixed d=0 and d=1 reported for every dataset | **Fixed: always report both d=0 and d=1**, flag in text if d=1 is still non-stationary, do not auto-difference further | Adaptive stopping would make the cross-domain table structurally different per dataset (some rows at d=1, others at d=2), breaking the like-for-like comparison RQ2 needs. A fixed, uniform depth keeps the table comparable; deeper differencing (if ever needed) is a modelling-stage decision, not an EDA one. |
| Decomposition method | Classical `seasonal_decompose` vs. STL | **STL (`robust=True`)** | Works uniformly across the daily (`period=7`) and weekly (`period=52`) datasets, and is robust to the kind of record-level noise already confirmed in the ED dataset ([reports/recovery_validation.txt](../reports/recovery_validation.txt)) — classical decomposition's trend/seasonal split is more easily distorted by outlier days. |
| Duration units in reports/plots | Keep each dataset's native unit (seconds/hours/minutes) vs. convert all to minutes for display | **Convert to minutes for all plots and the cross-domain table**, keep the per-dataset text report noting the original unit and conversion factor | The four datasets use three different native units (`aHT_Seconds`, `resolution_time_hours`, `*_minutes` ×2); a side-by-side RQ2 table in mixed units invites misreading. Conversion is presentation-only — it does not touch ADF/ACF, which are scale-invariant — so it carries no risk to the statistical results themselves. |
| Missing/invalid record handling | Re-derive independently vs. reuse `data_quality.py`'s exact drop rules (NaN/negative/zero duration dropped before series construction) | **Reuse the identical rule** (duplicated per the series-construction decision above, not re-derived) | A second, slightly different cleaning rule would mean EDA and the quality screen are silently looking at two different series for the "same" dataset — exactly the drift rule 1 exists to prevent for the quality gate, and the same logic applies here even though EDA isn't itself a gate. |

## CONSTRAINTS

- Operates only on datasets that are both (a) one of the four final
  datasets and (b) currently passing `data_quality.py`'s gate — a dataset
  failing the gate is skipped, not analyzed with caveats.
- Uses each dataset's Stage-0 config exactly as already committed — no
  new frequency/aggregation/column decisions are made at this stage.
- Writes all output under `reports/eda/`, never into `reports/` directly
  — keeps EDA artifacts separate from Stage-0 quality reports and the
  recovery-validation report.
- Does not modify `data_quality.py`, `recover_structure.py`,
  `ground_truth.json`, or any existing config/spec — purely additive.
- No ARIMA order selection or model fitting for the eventual baseline
  comparison happens here; this stage stops at identification diagnostics.
- Fully deterministic: no random sampling. Every number and plot must be
  exactly reproducible from the already-committed CSVs (no seed needed,
  but this must be stated and true).
- Uniform methodology across all qualifying datasets, so the cross-domain
  table is a fair comparison, not bespoke per-dataset analyses.

## RISKS

- **Unequal statistical power across domains.** The generated series have
  n=1,461 daily observations; `call_center_emea_2021_2025` has n=262
  weekly observations. ADF/ACF results must not be presented as equally
  reliable across domains without noting this — it's itself a relevant
  RQ2 observation (fewer data points is itself a "differing temporal
  property" some domains have).
- **Two datasets have no ground truth.** For `synthetic_it_support_tickets`
  and `call_center_emea_2021_2025`, findings are exploratory only — the
  report language must not imply anything was "recovered" or "validated"
  the way `reports/recovery_validation.txt` does for the synthetic pair.
- **Scope creep into model selection.** ACF/PACF eyeballing invites
  informally picking ARIMA orders during EDA. This stage's output is
  diagnostic only; order selection is explicitly deferred to a future
  "fit ARIMA" stage per `CLAUDE.md`'s pipeline description.
- **Unit-conversion errors** (seconds→minutes, hours→minutes) are a new
  source of transcription bugs not present in the existing pipeline —
  needs the conversion factor stated in both the per-dataset report and
  the code as a named constant, not inline.

## SUCCESS / ACCEPTANCE CRITERIA

- EDA runs only for datasets whose current `data_quality.py` verdict is
  `PASS`; a dataset that fails the gate produces no EDA output and is
  listed as skipped (with the failing gate reason) in the cross-domain
  summary rather than silently omitted.
- For each qualifying dataset: `reports/eda/<name>_eda.txt` plus the
  histogram, time-series+rolling-mean plot, seasonal boxplot, ACF/PACF,
  and STL-decomposition plots exist under `reports/eda/`, built strictly
  from that dataset's existing Stage-0 config.
- Each report states: record-level and series-level descriptive stats,
  the missingness/outlier recap sourced from `reports/<name>_quality.txt`,
  ADF result at d=0 and d=1, dominant ACF/PACF lag(s), and a
  plain-language seasonality/trend note.
- `reports/eda/eda_cross_domain_summary.txt` tabulates all qualifying
  datasets on identical columns (n, frequency, ADF p @ d=0/d=1, suggested
  d, dominant lag, CV, visible seasonality y/n).
- Re-running the stage against the committed CSVs reproduces identical
  numeric output.
- No exit code / pass-fail semantics of its own anywhere in this stage
  (it only reads, never sets, the Stage-0 verdict).
- Nothing written by this stage appears in `reports/` outside
  `reports/eda/`.

## TASKS

| # | Task | Depends on | Independently verifiable by |
|---|------|-----------|------------------------------|
| T1 | Finalize this spec's open parameters into a short reference table: per-dataset unit-conversion factor, STL `period`, day-of-week/month grouping for the seasonal boxplot, and the exact metric list that will appear in both the per-dataset report and the cross-domain table. | — | Table exists and covers all four datasets with no blanks. |
| T2 | Implement a quality-gate check helper: given a `configs/<name>.yaml`, determine whether that dataset currently passes `data_quality.py` (re-run it or read its latest `reports/<name>_quality.txt`), and skip/report accordingly. | — | Running it against each of the 4 configs correctly reports today's PASS status for all four. |
| T3 | Implement `src/eda.py`: for each dataset that clears T2's gate check, load + construct the series (duplicating `data_quality.py`'s rules, per the design-decisions table), compute basic EDA (descriptive stats, histogram, time series + rolling mean, seasonal boxplot) and diagnostic EDA (ADF(d=0), ADF(d=1), STL decomposition, ACF/PACF plots); write `reports/eda/<name>_eda.txt` + plots. | T1, T2 | Running against each of the 4 configs produces a report + plots under `reports/eda/` containing every T1-listed metric, and datasets failing T2's check produce no output. |
| T4 | Implement the cross-domain summary generator (collects each qualifying dataset's key metrics into one table, lists any skipped dataset with its failing-gate reason). | T3 | `reports/eda/eda_cross_domain_summary.txt` has one row per qualifying dataset, same columns, plus any skips noted. |
| T5 | Run EDA on the two ground-truth datasets and cross-check: does the ADF/stationarity conclusion here match `reports/recovery_validation.txt`'s ADF results for the same series? | T3, T4 | No contradiction between the two reports for the same dataset; any difference is explained (e.g. different d). |
| T6 | Run EDA on the two no-ground-truth datasets and write up domain-specific observations (native unit, missingness pattern, any visible structural break) as explicitly exploratory findings. | T3, T4 | Reports exist and each states "no ground truth available — findings are exploratory." |

**Ordering:** T1 → T2 → T3 → T4 → {T5, T6 in parallel}.

## OUT OF SCOPE

- ARIMA order selection / fitting for the baseline comparison (a separate,
  later pipeline stage per `CLAUDE.md`).
- EDA of the four screening-candidate configs that are not part of the
  final four (`AI_Workflow_Optimization_Dataset_2500_Rows_v1`,
  `Help_desk_tickets_issues_2015on`, `call_center_operations_wfo_dataset`,
  `hybrid_manufacturing_categorical`).
- Written dissertation-text interpretation of the cross-domain findings —
  this stage produces reproducible artifacts (reports, tables, plots);
  prose interpretation is the student's write-up, not pipeline code.
- Diebold–Mariano or any inter-model significance testing (already
  deferred pending supervisor confirmation per `CLAUDE.md`).
