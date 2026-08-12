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

## Rules

1. `data_quality.py` (with per-dataset YAML configs in `configs/`) is the ONLY
   quality screen. Never create any second quality script — dual thresholds 
   break the pre-registration claim.
2. `ground_truth.json` and `generate_datasets.py` must be committed
   BEFORE any ARIMA fitting (commit timestamp = pre-registration evidence).
3. `recover_structure.py` must validate recovered parameters against
   `ground_truth.json` before ARIMA fitting begins.
4. All generation is seeded with 30124470. Synthetic CSVs in `data/synthetic/`
   are committed for examiner verifiability — regeneration must reproduce them
   exactly. Never change the seed: the committed data and all recovery results
   depend on it.
5. Report honest results. Known/expected: ED dataset AR(1) φ=0.60 recovers at
   ~0.50 due to aggregation noise — this is a documented finding, not a bug.
   Do not "fix" it.
6. /strict: flag all Claude-drafted content, report changes honestly, no
   diplomatic softening.

## Evaluation

- Baselines: naïve, historical mean, moving average, exponential smoothing
- Metrics: MAE, RMSE, MAPE + significance testing
- Diebold–Mariano test: NOT confirmed by supervisor — do not implement as
  final methodology without confirmation