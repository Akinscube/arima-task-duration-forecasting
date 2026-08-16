# Spec: Model Evaluation (Metrics & Benchmarks)

## GOAL (restated)

Compare ARIMA's already-persisted test-set forecasts against each of the
four baselines' already-persisted test-set forecasts, per dataset, on
MAE/RMSE/MAPE — and determine, using the Diebold–Mariano test, whether
ARIMA's advantage or disadvantage on each dataset is statistically real —
so RQ1 and RQ2 have a defensible, reproducible verdict to draw on rather
than raw numbers alone.

**Status note (supersedes the original Wilcoxon plan below where they
conflict):** this stage was originally specced around a Wilcoxon
signed-rank test, because `CLAUDE.md` flags Diebold–Mariano as "NOT
confirmed by supervisor — do not implement as final methodology without
confirmation." A diagnostic run (Ljung-Box on the loss-differential series
for all 16 dataset/baseline pairs) found 5 pairs with statistically
significant serial autocorrelation — exactly what Wilcoxon cannot correct
for and what DM's HAC variance estimator exists to handle. On the
student's explicit instruction, this stage now uses DM instead, **while
supervisor confirmation is still pending** (message sent, not yet
answered as of this spec revision). Every report and output this stage
produces must say "provisional pending supervisor confirmation" — this
spec does not resolve or silently drop that caveat, and `CLAUDE.md`'s own
"not confirmed" line is deliberately left unedited.

## WHAT

For each of the four final datasets that has successful, non-skipped
output in **both** `reports/arima/<name>_arima_forecasts.csv` and
`reports/baselines/<name>_baselines_forecasts.csv`:

1. Load both CSVs; verify their test-window indices are identical (same
   dates) — a hard precondition, not silently reconciled (see
   Constraints/Risks).
2. For each of the four baselines (naïve, historical mean, moving
   average, SES), compute the paired **absolute-error differential**
   `d_t = |baseline_error_t| - |arima_error_t|` over the test window.
3. Run a **Diebold–Mariano test** on `d_t` for each of the four
   ARIMA-vs-baseline pairs: mean of `d_t`, tested against zero using a
   HAC (Newey–West) autocorrelation-robust standard error — implemented
   as an OLS regression of `d_t` on a constant with `cov_type="HAC"`
   (see Design Decisions and `model-evaluation-parameters.md` for the
   exact lag-selection rule). Also run and report the Ljung-Box test on
   `d_t` itself (the diagnostic that motivated this choice), for every
   pair, not just the 5 originally-flagged ones.
4. Apply **Holm–Bonferroni correction** within each dataset's family of 4
   tests (one family per dataset, not across all 16).
5. Classify each pair's outcome: **ARIMA wins** (ARIMA's MAE is lower AND
   Holm-adjusted p < α), **baseline wins** (baseline's MAE is lower AND
   Holm-adjusted p < α), or **no significant difference** (otherwise).
6. Write a per-dataset report (MAE/RMSE/MAPE table for ARIMA + all four
   baselines, raw and Holm-adjusted p-values, win/loss/tie verdict per
   baseline) and a cross-domain summary tabulating win/loss/tie counts per
   dataset.
7. Produce one cross-domain visualization: a dataset × baseline heatmap of
   the win/loss/tie verdicts (see Design Decisions — included from the
   start here, rather than added after the fact as with the previous two
   stages, since the same request has now come up twice running).

## WHY

- **RQ1** asks whether ARIMA beats the four named baselines on
  MAE/RMSE/MAPE. The metrics already exist in two separate stages'
  reports; this stage is what actually puts them side by side and answers
  the comparative question RQ1 poses.
- **CLAUDE.md**'s stated deliverable is baselines-vs-ARIMA comparison
  "with significance testing." A concrete diagnostic (Ljung-Box on all 16
  loss-differential series) found real serial autocorrelation in 5 pairs —
  the exact condition DM's HAC variance estimator exists to handle, and
  which a plain paired test (the originally-specced Wilcoxon) cannot
  correct for. That finding is the working justification for using DM now
  rather than waiting; supervisor confirmation has been requested but not
  yet received (see Status note above).
- **RQ2** needs, per dataset, not just "ARIMA's MAPE was higher/lower" but
  a real handle on *how much of that difference is signal vs. noise* —
  the win/loss/tie classification is what lets the later "analyse and
  interpret" stage say "ARIMA underperforms in domain X" with statistical
  backing rather than eyeballing a metrics table.

## DESIGN DECISIONS

| Decision | Options considered | Choice | Why |
|---|---|---|---|
| Significance test | Wilcoxon signed-rank (original spec) vs. paired t-test vs. Diebold–Mariano | **Diebold–Mariano** (mean loss differential, HAC/Newey–West standard error) | Superseded from Wilcoxon after the Ljung-Box diagnostic confirmed real serial autocorrelation in 5/16 loss-differential series — DM's HAC correction is designed for exactly that. Still provisional pending supervisor confirmation (see Status note); implemented on the student's explicit instruction, not a unilateral substitution. |
| DM implementation | Third-party `dm_test` package (adds a new dependency) vs. hand-rolled formula vs. OLS-on-constant with `cov_type="HAC"` | **OLS-on-constant with HAC covariance** (`statsmodels.api.OLS(d_t, const).fit(cov_type="HAC", cov_kwds={"maxlags": L})`) | Mathematically identical to the standard DM statistic (testing whether the mean of `d_t` is zero, with an autocorrelation-robust variance) but uses only `statsmodels`, already a project dependency — no new package to vet or justify. |
| HAC lag length `L` | Classical forecast-horizon rule (`L = h-1 = 0` for 1-step-ahead forecasts) vs. Newey–West automatic bandwidth selection | **Newey–West automatic bandwidth**, `L = floor(4*(n/100)^(2/9))` | The classical `h-1` rule assumes autocorrelation arises only from multi-step-forecast overlap, which doesn't apply here (`h=1`). The autocorrelation this stage actually observed comes from baseline model misspecification (e.g. historical-mean's slow-moving error), not forecast-horizon theory — an automatic, data-driven bandwidth is the honest choice given that mismatch, not the textbook default for a different situation. |
| Loss function tested | Absolute error (aligns with MAE) vs. squared error (aligns with RMSE) vs. both | **Absolute error only** | Testing both would triple the test count (48 instead of 16) for a single stage without a corresponding request to report RMSE-specific significance. MAE is RQ1's first-listed metric; RMSE and MAPE are still reported descriptively per pair, just not separately significance-tested. Documented as a scope limit, not hidden. |
| "Beats" definition | Point-estimate only vs. requires significance | **Requires both a lower point-estimate AND Holm-adjusted p < α** | A lower point estimate on a modest test set (e.g. `call_center_emea_2021_2025`'s n_test=53) can be noise; gating the win/loss table on significance keeps RQ1's conclusions defensible rather than overclaiming from a metrics table alone. |
| Multiple-comparisons correction | None vs. Holm–Bonferroni per-dataset family (4 tests) vs. Benjamini–Hochberg across all 16 | **Holm–Bonferroni within each dataset's 4-test family** | Running 16 tests at raw α=0.05 risks roughly one false positive by chance alone. Correcting per-dataset (not across all 16) respects the actual comparison structure RQ1/RQ2 care about — "does ARIMA beat baselines within this domain" — rather than treating all four domains as one undifferentiated pool. |
| Ljung-Box reporting scope | Only the 5 originally-flagged pairs vs. all 16 pairs | **All 16, every run** | Now that DM is the test, Ljung-Box's role shifts from "diagnosing a Wilcoxon problem" to "documenting why HAC correction is warranted, per pair" — worth having for every pair as standing evidence, not just the ones that happened to be flagged in one manual diagnostic run. |
| Cross-domain visualization | Add now vs. wait to be asked (as with the previous two stages) | **Add now**: a dataset × baseline win/loss/tie heatmap | The same follow-up request ("add a visualization") has now happened after both prior stages' initial implementations. Including it in this stage's original scope avoids a third round of the same amendment cycle — flagged here explicitly as a proactive inclusion, not silently assumed. |
| Data source | Recompute forecasts vs. reuse each stage's already-persisted per-point CSVs | **Reuse only** (`reports/arima/<name>_arima_forecasts.csv`, `reports/baselines/<name>_baselines_forecasts.csv`) | This is exactly what both prior stages' specs cited as the reason for persisting per-point forecasts/errors in the first place — this stage doing anything else would waste that design work and risk numeric drift from two different code paths computing "the same" forecasts. |

## CONSTRAINTS

- Reads only the already-persisted `reports/arima/<name>_arima_forecasts.csv`
  and `reports/baselines/<name>_baselines_forecasts.csv` — never refits or
  regenerates any forecast.
- Evaluates a dataset only if it has successful (non-skipped, non-failed)
  output in **both** source CSVs; otherwise the dataset is skipped in this
  stage's summary with a stated reason.
- Requires the two CSVs' test-window indices to be identical (same dates,
  same count). A mismatch is a reportable failure for that dataset, not
  silently reindexed, truncated, or reconciled.
- Significance test: Diebold–Mariano, implemented as OLS-on-constant with
  HAC (Newey–West) covariance on `|baseline_error| - |arima_error|` per
  test point, per baseline, per dataset (exact call in
  `model-evaluation-parameters.md`).
- Every report and output this stage produces states, verbatim,
  "Diebold–Mariano — provisional pending supervisor confirmation" — this
  is a hard requirement, not a suggestion, given `CLAUDE.md`'s unresolved
  caveat.
- Correction: Holm–Bonferroni, applied within each dataset's 4-test family
  independently (not pooled across datasets).
- α = 0.05 for both the raw and Holm-adjusted significance threshold
  (matches the ADF α already used throughout this pipeline).
- Ljung-Box (on `d_t`) is computed and reported for all 16 pairs, not only
  as a one-off diagnostic — it's the standing evidence for why HAC
  correction is warranted per pair.
- `CLAUDE.md`'s "Diebold–Mariano... NOT confirmed by supervisor" line is
  not edited by this stage — it stays accurate until confirmation actually
  arrives.
- Output under `reports/evaluation/`, never into `reports/`,
  `reports/eda/`, `reports/baselines/`, or `reports/arima/`.
- Purely additive — does not modify `data_quality.py`, `eda.py`,
  `baselines.py`, `arima_model.py`, `recover_structure.py`,
  `ground_truth.json`, or any spec/config from a prior stage.

## RISKS

- **This stage's DM-based verdicts are provisional and could be
  retracted or revised.** Supervisor confirmation has been requested but
  not received. If the supervisor declines DM, requires a different HAC
  lag rule, or wants a different loss function, every win/loss verdict in
  this stage may need re-deriving. This is the central risk of proceeding
  now rather than waiting — stated plainly, not softened, per every
  report carrying the "provisional" label.
- **HAC lag-length choice affects the p-value.** The Newey–West automatic
  bandwidth is a defensible default, not the only reasonable one; a
  different `L` could shift a borderline pair (e.g. anything landing near
  the Holm-adjusted α threshold) from significant to not, or vice versa.
  Must be stated as a methodological choice in every report, not
  presented as if the p-value were unique/exact.
- **DM is an asymptotic test; modest test sets weaken it.**
  `call_center_emea_2021_2025` has only 53 test points — both the
  point-estimate of the mean loss differential and its HAC variance
  estimate are less reliable there than on the ~290-point daily datasets.
  A "no significant difference" result on that dataset may reflect
  insufficient data, not genuine equivalence — must be stated as a power
  limitation, not overclaimed as "ARIMA ties this baseline."
- **`generated_job_shop_manufacturing`'s naïve and SES baselines are
  numerically identical** (SES's fitted α converged to 1.0, per the
  baseline stage's own report), so the ARIMA-vs-naïve and ARIMA-vs-SES DM
  tests on that dataset will produce identical statistics and p-values.
  Expected consequence of an already-documented baseline-stage finding,
  not a bug here — must be noted in the per-dataset report.
- **MAE-only significance testing** means a dataset could show, e.g., a
  significant MAE advantage but an ambiguous or reversed RMSE/MAPE
  picture; both are still reported (descriptively) alongside the MAE-based
  verdict so this asymmetry is visible, not concealed by only reporting
  the metric that was tested.
- **Per-dataset (not global) correction** means this stage does not
  control the false-discovery rate across all 16 tests jointly — a
  cross-domain claim like "ARIMA beat baselines in 3 of 4 domains" is
  still an informal aggregate of four independently-corrected families,
  not itself a single corrected statistic. Flagged for the later
  "analyse and interpret results" stage to handle carefully, not silently
  treated as one big controlled experiment here.

## SUCCESS / ACCEPTANCE CRITERIA

- Runs only for datasets with successful output present in both prior
  stages' CSVs; any dataset missing from either (or index-mismatched) is
  skipped with a stated reason in the cross-domain summary.
- Per-dataset report (`reports/evaluation/<name>_evaluation.txt`) shows:
  MAE/RMSE/MAPE for ARIMA and all four baselines side by side; for each of
  the four ARIMA-vs-baseline pairs, the Ljung-Box result on `d_t`, the raw
  DM p-value, the Holm-adjusted p-value, and the win/loss/tie verdict; the
  "provisional pending supervisor confirmation" label; and the
  job-shop-naïve/SES-identical-test note where applicable.
- `reports/evaluation/evaluation_cross_domain_summary.txt` tabulates, per
  dataset, counts of ARIMA wins / baseline wins / ties across the four
  baseline comparisons.
- `reports/evaluation/evaluation_win_loss_heatmap.png` exists: dataset ×
  baseline grid, colored by verdict.
- Re-running end-to-end reproduces identical output (no stochastic fitting
  in this stage — the significance test is deterministic given its fixed
  inputs, so exact reproducibility is expected and must be verified, not
  merely assumed).
- Every design decision above (loss function choice, HAC lag rule,
  correction method, "beats" definition, provisional DM status) is stated
  in the per-dataset report text itself, not only implied by the numbers.

## TASKS

| # | Task | Depends on | Independently verifiable by |
|---|------|-----------|------------------------------|
| T1 | Write `model-evaluation-parameters.md`: α, loss-differential formula, exact DM/HAC OLS call signature and lag-selection rule, Ljung-Box call, Holm–Bonferroni family definition, win/loss/tie classification rule, output paths. | — | Doc exists, no blanks, every constant traces to a stated rationale. |
| T2 | Implement the loader/join: read both source CSVs per dataset, verify identical test-window indices, build the four paired absolute-error-differential arrays. | T1 | Running against all four datasets' existing CSVs reports matching index lengths/dates for each; a deliberately truncated copy of one CSV (in a scratch test) is correctly reported as a mismatch, not silently handled. |
| T3 | Implement the per-pair Ljung-Box test, the DM test (OLS-on-constant with HAC covariance), and Holm–Bonferroni correction within each dataset's 4-test family. | T2 | Against a hand-built synthetic example (ARIMA's absolute errors uniformly smaller by a fixed margin across all test points, for all four baselines), the raw p-values are very small and the win verdict is "ARIMA wins" for all four after correction. Cross-checked against this stage's own earlier scratch diagnostic (`check_dt_autocorrelation.py`'s Ljung-Box numbers) for the real 16 pairs — must match exactly. |
| T4 | Implement the win/loss/tie classification per pair per the locked rule (lower MAE AND Holm-adjusted p < α). | T3 | On the same synthetic fixture, plus a second fixture with identical ARIMA/baseline errors (no real difference), the second fixture classifies as "no significant difference" for all four pairs. |
| T5 | Implement the per-dataset report and cross-domain summary. | T4 | Reports exist under `reports/evaluation/` with all T1-listed content; cross-domain summary has one row per evaluated dataset with correct win/loss/tie counts. |
| T6 | Implement the win/loss/tie heatmap. | T5 | Plot exists, one cell per (dataset, baseline) pair, visually matches the text verdicts in each per-dataset report. |
| T7 | Run end-to-end against all four final datasets; verify reproducibility by re-running and diffing output. | T2–T6 | Two consecutive runs produce byte-identical output. |

**Ordering:** T1 → T2 → T3 → T4 → T5 → T6 → T7.

## OUT OF SCOPE

- Wilcoxon signed-rank testing (the originally-specced approach,
  superseded — see Status note. Could still be reported as a secondary
  cross-check if wanted later, but is not part of this revision's scope).
- Editing `CLAUDE.md`'s "not confirmed by supervisor" line — stays as-is
  until confirmation actually arrives, regardless of what this stage
  implements in the meantime.
- Significance testing on RMSE (squared-error loss) or MAPE — only the
  MAE-aligned absolute-error differential is tested; RMSE/MAPE are
  reported descriptively per pair.
- Correction pooling all 16 tests into one family (locked as per-dataset
  families of 4).
- Refitting or regenerating any forecast from the baseline or ARIMA
  stages — this stage is read-only with respect to both.
- Full narrative interpretation of what the win/loss pattern means for
  RQ1/RQ2 across all four domains — that is the later "analyse and
  interpret results" stage; this stage produces the verdicts it will
  draw on.
