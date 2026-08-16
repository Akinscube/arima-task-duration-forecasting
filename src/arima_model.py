"""
ARIMA fitting and tuning (main model) — see spec/arima-fitting.md and
spec/arima-parameters.md.

Runs only against datasets that currently pass src/data_quality.py's
quality gate (re-checked at run time, same precondition src/eda.py and
src/baselines.py use). For each qualifying dataset:

  1. Splits the series into train/test using data_quality.py's Gate 3 rule
     (TRAIN_FRACTION=0.8), identical to src/baselines.py.
  2. Selects d via ADF (alpha=0.05) on the TRAINING split only (not EDA's
     full-series suggested_d — see spec/arima-fitting.md's design
     decisions on why the two may differ).
  3. Bounds candidate p/q via ACF/PACF significance (+/-1.96/sqrt(n_train))
     on the training-differenced series, capped at P_MAX/Q_MAX.
  4. Grid-searches (p, d_fixed, q), scores by AIC (BIC reported alongside),
     excludes non-converged/non-stationary/non-invertible fits.
  5. Fits once at the selected order on the full training set, then
     produces walk-forward one-step-ahead test forecasts by extending the
     fitted model's state with each revealed actual
     (results.append(..., refit=False)) — coefficients are never
     re-estimated during the test phase, mirroring src/baselines.py's SES
     convention.
  6. Computes MAE/RMSE/MAPE and persists per-point forecasts/errors.

Output goes to reports/arima/ — kept separate from reports/, reports/eda/,
reports/baselines/, and the recovery-validation report. This stage does
NOT compare against baselines or run significance testing (Diebold-Mariano
remains unconfirmed per CLAUDE.md) — that is a later stage.

Usage:
    python src/arima_model.py                          # the four final datasets
    python src/arima_model.py configs/<name>.yaml [...] # explicit subset
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import data_quality  # reuse the canonical quality gate, not reimplement it

FINAL_FOUR = [
    "configs/synthetic_it_support_tickets.yaml",
    "configs/call_center_emea_2021_2025.yaml",
    "configs/generated_emergency_dept_stays.yaml",
    "configs/generated_job_shop_manufacturing.yaml",
]

# spec/arima-parameters.md
TRAIN_FRACTION = 0.8  # matches data_quality.py's Gate 3 rule / baselines.py
ADF_ALPHA = 0.05
P_MAX = 5
Q_MAX = 5
MAPE_ZERO_EPS = 1e-9

OUT_DIR = Path("reports/arima")


def quality_gate_passed(cfg_path: str) -> tuple[bool, dict, str]:
    """Re-run data_quality.py's own profiler — the single source of truth
    for the gate (rule 1) — rather than reading a possibly stale report."""
    cfg = data_quality.load_config(cfg_path)
    lines, outcome = data_quality.profile(cfg)
    report = "\n".join(lines)
    Path("reports").mkdir(exist_ok=True)
    out_path = f"reports/{outcome['name']}_quality.txt"
    Path(out_path).write_text(report)
    return outcome["passed"], cfg, out_path


def build_series(cfg: dict) -> pd.Series:
    """Construct the aggregated series exactly as data_quality.py does
    (same drop rules, same resample/aggregate) — duplicated rather than
    imported, matching recover_structure.py's/eda.py's/baselines.py's
    precedent."""
    df = pd.read_csv(cfg["file"])
    ts_col = cfg["timestamp_column"]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", format="ISO8601")

    if "duration_column" in cfg:
        df["duration"] = pd.to_numeric(df[cfg["duration_column"]], errors="coerce")
    else:
        end = pd.to_datetime(df[cfg["end_column"]], errors="coerce", format="ISO8601")
        df["duration"] = (end - df[ts_col]).dt.total_seconds() / 60.0

    df = df.dropna(subset=[ts_col])
    df = df[(df["duration"] > 0)].dropna(subset=["duration"])

    freq = cfg.get("frequency", "D")
    agg = cfg.get("aggregation", "median")
    series = df.set_index(ts_col)["duration"].resample(freq).agg(agg).asfreq(freq)
    return series


def train_test_split(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Identical split rule to baselines.py: contiguous, time-ordered,
    first n_train / last n_test positions (0%-missing precondition holds
    for all four final datasets, per spec/baseline-parameters.md)."""
    n = len(series)
    n_train = int(n * TRAIN_FRACTION)
    return series.iloc[:n_train], series.iloc[n_train:]


def select_d(train: pd.Series) -> dict:
    """ADF (alpha=0.05) on the TRAINING split only — deliberately not
    EDA's full-series suggested_d, to keep order selection lookahead-free
    (spec/arima-fitting.md design decision: 'd selection')."""
    p_d0 = float(adfuller(train.dropna())[1])
    train_diff1 = train.diff().dropna()
    p_d1 = float(adfuller(train_diff1)[1])

    if p_d0 <= ADF_ALPHA:
        d, resolved = 0, True
    elif p_d1 <= ADF_ALPHA:
        d, resolved = 1, True
    else:
        d, resolved = None, False

    return {"d": d, "resolved": resolved, "adf_p_d0": p_d0, "adf_p_d1": p_d1}


def acf_pacf_bounds(train: pd.Series, d: int) -> dict:
    """Bound candidate p/q via ACF/PACF significance on the training
    series differenced d times, capped at P_MAX/Q_MAX
    (spec/arima-parameters.md)."""
    stationary = train.dropna() if d == 0 else train.diff().dropna()

    n_train = len(train)
    nlags = min(40, n_train // 2 - 1)
    nlags = min(nlags, len(stationary) - 1)  # statsmodels cannot exceed series length

    band = 1.96 / np.sqrt(n_train)
    acf_vals = acf(stationary, nlags=nlags)[1:]  # drop lag 0
    pacf_vals = pacf(stationary, nlags=nlags)[1:]

    acf_sig_lags = [lag for lag, v in enumerate(acf_vals, start=1) if abs(v) > band]
    pacf_sig_lags = [lag for lag, v in enumerate(pacf_vals, start=1) if abs(v) > band]

    q_max = min(max(acf_sig_lags), Q_MAX) if acf_sig_lags else 0
    p_max = min(max(pacf_sig_lags), P_MAX) if pacf_sig_lags else 0

    return {
        "nlags": nlags,
        "band": band,
        "acf_sig_lags": acf_sig_lags,
        "pacf_sig_lags": pacf_sig_lags,
        "p_max": p_max,
        "q_max": q_max,
    }


def grid_search(train: pd.Series, d: int, p_max: int, q_max: int) -> list[dict]:
    """Fit every (p, d_fixed, q) candidate; exclude non-converged or
    non-stationary/non-invertible fits with a logged reason
    (spec/arima-parameters.md)."""
    trend = "c" if d == 0 else "n"
    candidates = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            order = (p, d, q)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = ARIMA(train, order=order, trend=trend,
                                   enforce_stationarity=True, enforce_invertibility=True)
                    result = model.fit()
                    non_converged = any(issubclass(w.category, ConvergenceWarning) for w in caught)
                if non_converged:
                    candidates.append({"order": order, "converged": False,
                                        "reason": "did not converge (ConvergenceWarning)"})
                else:
                    candidates.append({"order": order, "converged": True,
                                        "aic": float(result.aic), "bic": float(result.bic)})
            except Exception as e:
                candidates.append({"order": order, "converged": False,
                                    "reason": f"exception: {type(e).__name__}: {e}"})
    return candidates


def select_order(candidates: list[dict]) -> dict:
    converged = [c for c in candidates if c["converged"]]
    if not converged:
        return {"selected_order": None, "aic_bic_agree": None}
    best_aic = min(converged, key=lambda c: c["aic"])
    best_bic = min(converged, key=lambda c: c["bic"])
    return {
        "selected_order": best_aic["order"],
        "selected_aic": best_aic["aic"],
        "selected_bic": best_aic["bic"],
        "bic_preferred_order": best_bic["order"],
        "aic_bic_agree": best_aic["order"] == best_bic["order"],
    }


def walk_forward_forecast(train: pd.Series, test: pd.Series, order: tuple, trend: str) -> pd.Series:
    """Fit once on train; extend state only via append(refit=False) for
    each revealed test actual — order and coefficients are frozen for the
    whole test window (spec/arima-fitting.md: 'Walk-forward refit')."""
    model = ARIMA(train, order=order, trend=trend,
                   enforce_stationarity=True, enforce_invertibility=True)
    results = model.fit()

    forecasts = []
    for ts, actual in zip(test.index, test.values):
        fc = float(results.forecast(1).iloc[0])
        forecasts.append(fc)
        # name must match train's column name, or statsmodels' internal
        # concat (append) raises "Columns must match to concatenate"
        new_obs = pd.Series([actual], index=[ts], name=train.name)
        results = results.append(new_obs, refit=False)

    return pd.Series(forecasts, index=test.index)


def compute_metrics(actual: pd.Series, forecast: pd.Series) -> tuple[dict, pd.Series]:
    """Identical convention to baselines.py: error = actual - forecast,
    MAPE excludes |actual| <= MAPE_ZERO_EPS."""
    error = actual - forecast
    abs_error = error.abs()
    mae = float(abs_error.mean())
    rmse = float(np.sqrt((error ** 2).mean()))

    nonzero = actual.abs() > MAPE_ZERO_EPS
    n_excluded = int((~nonzero).sum())
    mape = float((abs_error[nonzero] / actual[nonzero].abs()).mean() * 100) if nonzero.any() else float("nan")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4) if not np.isnan(mape) else "nan",
        "mape_excluded_zero_actuals": n_excluded,
    }, error


def plot_forecast(name: str, cfg: dict, test: pd.Series, forecast: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(test.index, test.values, label="actual", color="black", lw=1.5)
    ax.plot(test.index, forecast.values, label="arima", color="tab:blue", lw=1, alpha=0.9)
    unit = cfg.get("duration_column", "duration")
    ax.set_title(f"{name} — test-period actual vs. ARIMA forecast ({unit})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_arima_forecast_plot.png", dpi=150)
    plt.close(fig)


def run_arima(name: str, cfg: dict) -> dict:
    """Returns a result dict with either a populated 'row' (success) or a
    'failure' reason (d unresolved / no convergent candidate) — the
    caller reports failures explicitly rather than silently dropping the
    dataset (spec/arima-fitting.md's non-convergence / d>=2 risks)."""
    series = build_series(cfg)
    train, test = train_test_split(series)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"ARIMA — {name}", "=" * 60]
    lines.append(f"n_train: {len(train)}   n_test: {len(test)}")
    lines.append(f"train/test split: TRAIN_FRACTION={TRAIN_FRACTION} (data_quality.py Gate 3 rule)")
    lines.append("protocol: walk-forward, one-step-ahead, fit once on train "
                  "then extend state only (no refitting) — see spec/arima-fitting.md")
    lines.append("")

    # -------- [1] d selection (ADF, train-only)
    d_info = select_d(train)
    lines.append("[1] ORDER SELECTION — d (ADF, train-only, alpha=%.2f)" % ADF_ALPHA)
    lines.append(f"    d=0: p={d_info['adf_p_d0']:.4f}  "
                 f"({'stationary' if d_info['adf_p_d0'] <= ADF_ALPHA else 'non-stationary'})")
    lines.append(f"    d=1: p={d_info['adf_p_d1']:.4f}  "
                 f"({'stationary' if d_info['adf_p_d1'] <= ADF_ALPHA else 'non-stationary'})")
    if not d_info["resolved"]:
        lines.append("    d>=2 (not resolved) — this pipeline does not auto-difference "
                     "past d=1; ARIMA fitting stage FAILED for this dataset.")
        Path(OUT_DIR / f"{name}_arima.txt").write_text("\n".join(lines))
        return {"dataset": name, "failed": True, "reason": "d>=2 (not resolved)"}
    d = d_info["d"]
    lines.append(f"    selected d: {d}")
    lines.append("")

    # -------- [2] p/q bounding (ACF/PACF significance, train-only)
    bounds = acf_pacf_bounds(train, d)
    lines.append(f"[2] ORDER SELECTION — p/q bounds (ACF/PACF significance, "
                 f"band=+/-{bounds['band']:.4f}, nlags={bounds['nlags']})")
    lines.append(f"    PACF significant lags: {bounds['pacf_sig_lags'] or '(none)'}  -> p_max={bounds['p_max']}")
    lines.append(f"    ACF significant lags:  {bounds['acf_sig_lags'] or '(none)'}  -> q_max={bounds['q_max']}")
    lines.append("")

    # -------- [3] grid search (AIC/BIC)
    candidates = grid_search(train, d, bounds["p_max"], bounds["q_max"])
    lines.append(f"[3] ORDER SELECTION — grid search "
                 f"({len(candidates)} candidates, p in 0..{bounds['p_max']}, q in 0..{bounds['q_max']})")
    for c in candidates:
        if c["converged"]:
            lines.append(f"    {c['order']}: AIC={c['aic']:.2f}  BIC={c['bic']:.2f}")
        else:
            lines.append(f"    {c['order']}: EXCLUDED — {c['reason']}")

    selection = select_order(candidates)
    if selection["selected_order"] is None:
        lines.append("")
        lines.append("    ALL CANDIDATES EXCLUDED — ARIMA fitting stage FAILED for this dataset.")
        Path(OUT_DIR / f"{name}_arima.txt").write_text("\n".join(lines))
        return {"dataset": name, "failed": True, "reason": "no convergent candidate"}

    order = selection["selected_order"]
    lines.append("")
    lines.append(f"    selected order (AIC-preferred): {order}  "
                 f"AIC={selection['selected_aic']:.2f}  BIC={selection['selected_bic']:.2f}")
    if not selection["aic_bic_agree"]:
        lines.append(f"    NOTE: BIC prefers a different order: {selection['bic_preferred_order']} "
                     f"— AIC's pick is used (see spec/arima-fitting.md)")
    lines.append("")

    # -------- [4] final fit + walk-forward forecast
    trend = "c" if d == 0 else "n"
    forecast = walk_forward_forecast(train, test, order, trend)
    metrics, error = compute_metrics(test, forecast)

    out_df = pd.DataFrame({"actual": test, "arima_forecast": forecast, "arima_error": error})
    out_df.to_csv(OUT_DIR / f"{name}_arima_forecasts.csv")
    plot_forecast(name, cfg, test, forecast)

    lines.append(f"[4] FINAL MODEL: ARIMA{order}, trend='{trend}'")
    lines.append(f"    MAE={metrics['mae']}  RMSE={metrics['rmse']}  MAPE={metrics['mape']}%  "
                 f"(excluded {metrics['mape_excluded_zero_actuals']} zero-actual points)")
    lines.append("")
    lines.append(f"[output] per-point forecasts written to {OUT_DIR}/{name}_arima_forecasts.csv")
    lines.append(f"[output] actual-vs-forecast plot written to {OUT_DIR}/{name}_arima_forecast_plot.png")
    Path(OUT_DIR / f"{name}_arima.txt").write_text("\n".join(lines))

    return {
        "dataset": name,
        "failed": False,
        "n_train": len(train),
        "n_test": len(test),
        "d": d,
        "p_max": bounds["p_max"],
        "q_max": bounds["q_max"],
        "order": str(order),
        "aic_bic_agree": selection["aic_bic_agree"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
    }


def plot_mape_comparison(rows: list[dict]) -> None:
    """Cross-domain MAPE bar chart — MAPE, not MAE/RMSE, because
    build_series() leaves durations in each dataset's native unit
    (hours/seconds/minutes); MAPE is unit-independent so it's the one
    metric safe to compare across datasets (mirrors baselines.py's
    identical chart/rationale)."""
    if not rows:
        return
    datasets = [r["dataset"] for r in rows]
    mapes = [r["mape"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(datasets, mapes, color="tab:blue")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("ARIMA MAPE by dataset")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "arima_mape_comparison.png", dpi=150)
    plt.close(fig)


def write_cross_domain_summary(rows: list[dict], failed: list[dict], skipped: list[tuple[str, str]]) -> None:
    lines = ["ARIMA CROSS-DOMAIN SUMMARY", "=" * 60]
    lines.append("plain ARIMA(p,d,q), no seasonal term, fit once on train and extended "
                 "via append(refit=False) for walk-forward one-step-ahead test forecasts "
                 "(see spec/arima-fitting.md). No comparison against baselines or "
                 "significance testing in this stage.")
    lines.append("")
    if rows:
        cols = list(rows[0].keys())
        lines.append(" | ".join(cols))
        for row in rows:
            lines.append(" | ".join(str(row[c]) for c in cols))
    else:
        lines.append("(no qualifying/successful datasets)")
    if failed:
        lines.append("")
        lines.append("FAILED (order selection could not resolve):")
        for f in failed:
            lines.append(f"  {f['dataset']}: {f['reason']} — see {OUT_DIR}/{f['dataset']}_arima.txt")
    if skipped:
        lines.append("")
        lines.append("SKIPPED (failed quality gate):")
        for name, report_path in skipped:
            lines.append(f"  {name}: see {report_path}")
    lines.append("")
    for row in rows:
        lines.append(json.dumps(row))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR / "arima_cross_domain_summary.txt").write_text("\n".join(lines))


def main(argv: list[str]) -> int:
    cfg_paths = argv[1:] if len(argv) > 1 else FINAL_FOUR

    rows, failed, skipped = [], [], []
    for cfg_path in cfg_paths:
        passed, cfg, report_path = quality_gate_passed(cfg_path)
        name = cfg.get("name", Path(cfg_path).stem)
        if not passed:
            skipped.append((name, report_path))
            print(f"{name}: SKIPPED (fails quality gate, see {report_path})")
            continue
        result = run_arima(name, cfg)
        if result["failed"]:
            failed.append(result)
            print(f"{name}: FAILED ({result['reason']}), see {OUT_DIR}/{name}_arima.txt")
            continue
        rows.append(result)
        print(f"{name}: ARIMA{result['order']} written to {OUT_DIR}/{name}_arima.txt")

    write_cross_domain_summary(rows, failed, skipped)
    plot_mape_comparison(rows)
    print(f"\ncross-domain summary written to {OUT_DIR}/arima_cross_domain_summary.txt")
    print(f"cross-domain MAPE plot written to {OUT_DIR}/arima_mape_comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
