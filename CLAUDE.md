# CLAUDE.md — arima-task-duration-forecasting

MSc AI dissertation artefact (University of South Wales). Researcher: Akin.

**Project title (proposal — authoritative):** Evaluating ARIMA-Based Time
Series Forecasting for Task Duration Prediction in High-Volume Operational
Workflows.

## Goals

**Aim:** Evaluate whether ARIMA can accurately predict task durations in
high-volume operational workflows, and whether it beats simpler baselines —
using four structurally analogous domains as the testbed. Aviation MRO is
motivating background only; the framing everywhere is "high-volume operational
workflows" / "operational domains".

**Research questions (locked wording — never paraphrase):**

- **RQ1:** 1. Is the prediction of the task duration made using an ARIMA model more accurate than that done using conventional baseline methods of forecast (naïve, historical mean, moving average, exponential smoothing), as measured using MAE, RMSE, and MAPE when considering test data taken from different operational domains?
- **RQ2:** 2. How varied is the performance of the ARIMA forecasting technique between domains that have differing temporal properties? In what domain task duration times series is this technique underperforming, and how will this impact the use of ARIMA in workforce planning domains like aviation MRO?

**What the code must deliver:** a reproducible pipeline — generate/screen data
→ validate parameter recovery → fit ARIMA (ADF, ACF/PACF, AIC/BIC) → compare
against all four baselines on MAE/RMSE/MAPE with significance testing — such
that an examiner can rerun everything from the fixed seed and get identical
results. Every implementation choice should trace back to RQ1 or RQ2.
 
**Design stance:** the multi-domain setup (differing temporal properties
across the four datasets) is a deliberate design choice to answer RQ2, not a
data-availability workaround. ARIMA is argued on problem-type suitability,
never universal superiority over deep learning.
 
**Final four datasets:** synthetic_it_support_tickets (third-party Kaggle
synthetic), call centre (real Kaggle), generated_emergency_dept_stays and
generated_job_shop_manufacturing (researcher-generated). Other configs/CSVs in
the repo are screening candidates, not part of the final four.
 
## Rules
 
1. `src/data_quality.py` (with per-dataset YAML configs in `configs/`) is the
   ONLY quality screen. Never create a second quality script — dual thresholds
   break the pre-registration claim. Thresholds are documented in
   `spec/quality-gate-thresholds.md`; the script and spec must agree.
2. `ground_truth.json` and `src/generation/generate_datasets.py` must be
   committed BEFORE any ARIMA fitting (commit timestamp = pre-registration
   evidence).
3. `src/recover_structure.py` must validate recovered parameters against
   `ground_truth.json` before ARIMA fitting begins. Results go to
   `reports/recovery_validation.txt`.
4. All generation is seeded with 30124470. Generated CSVs in `data/raw/`
   (generated_*.csv) are committed for examiner verifiability — regeneration
   must reproduce them exactly. Never change the seed: the committed data and
   all recovery results depend on it.
5. Recovery pass/fail is judged ONLY against the per-dataset TOLERANCES
   declared in `src/recover_structure.py`. AR-coefficient attenuation under
   record noise + daily-median aggregation is an expected, reported outcome,
   not a failure. Out-of-tolerance results are documented, explained
   exceptions, not silently passed and not silently "fixed".
   Retuning data-generating process parameters is permitted only before ARIMA 
   fitting, and every change must be logged in `ground_truth.json` with rationale.
6. /strict: flag all Claude-drafted content, report changes honestly, no
   diplomatic softening.
## Evaluation
 
- Baselines: naïve, historical mean, moving average, exponential smoothing
- Metrics: MAE, RMSE, MAPE + significance testing
- Diebold–Mariano test: NOT confirmed by supervisor — do not implement as
  final methodology without confirmation