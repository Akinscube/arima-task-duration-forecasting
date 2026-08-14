"""
Baseline forecasting models — see spec/baseline-forecasting-models.md and
spec/baseline-parameters.md.

Runs only against datasets that currently pass src/data_quality.py's
quality gate (re-checked at run time, same precondition src/eda.py uses).
For each qualifying dataset, splits the series into train/test using
data_quality.py's Gate 3 rule (TRAIN_FRACTION=0.8), then produces
walk-forward one-step-ahead forecasts for four baselines — naive,
historical mean, moving average, simple exponential smoothing — computes
MAE/RMSE/MAPE for each, and persists per-test-point forecasts/errors so a
future evaluation stage can do significance testing without recomputing
baselines.

Output goes to reports/baselines/ — kept separate from reports/,
reports/eda/, and the recovery-validation report.

Usage:
    python src/baselines.py                          # the four final datasets
    python src/baselines.py configs/<name>.yaml [...] # explicit subset
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

import data_quality  # reuse the canonical quality gate, not reimplement it

FINAL_FOUR = [
    "configs/synthetic_it_support_tickets.yaml",
    "configs/call_center_emea_2021_2025.yaml",
    "configs/generated_emergency_dept_stays.yaml",
    "configs/generated_job_shop_manufacturing.yaml",
]

# spec/baseline-parameters.md
TRAIN_FRACTION = 0.8  # matches data_quality.py's Gate 3 rule
MA_WINDOW = 7
MAPE_ZERO_EPS = 1e-9
METHODS = ["naive", "historical_mean", "moving_average", "ses"]

OUT_DIR = Path("reports/baselines")


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
    imported, matching recover_structure.py's and eda.py's precedent."""
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
    """Contiguous, time-ordered split: first n_train / last n_test
    positions. Equivalent to data_quality.py's Gate 3 count-based
    n_train/n_test as long as the series has 0% missing periods (true for
    all four final datasets today — see spec/baseline-parameters.md)."""
    n = len(series)
    n_train = int(n * TRAIN_FRACTION)
    return series.iloc[:n_train], series.iloc[n_train:]


def naive_forecast(series: pd.Series) -> pd.Series:
    return series.shift(1)


def historical_mean_forecast(series: pd.Series) -> pd.Series:
    return series.expanding().mean().shift(1)


def moving_average_forecast(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean().shift(1)


def ses_forecast(train: pd.Series, test: pd.Series) -> tuple[pd.Series, float]:
    """Walk-forward one-step-ahead SES: alpha is fit once on train; the
    smoothed level then continues updating with each revealed test
    actual, without re-fitting alpha."""
    fit = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    alpha = float(fit.params["smoothing_level"])

    level = float(fit.forecast(1).iloc[0])  # forecast for first test point, train-only
    forecasts = []
    for actual in test.values:
        forecasts.append(level)
        level = alpha * float(actual) + (1 - alpha) * level

    return pd.Series(forecasts, index=test.index), alpha


def compute_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
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


def plot_forecasts(name: str, cfg: dict, test: pd.Series, forecasts: dict) -> None:
    """Actual vs. each method's forecast, over the test window, in the
    dataset's native duration unit (see design-decisions table for why
    the cross-domain chart uses MAPE instead of trying to compare native
    units across datasets)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(test.index, test.values, label="actual", color="black", lw=1.5)
    for method in METHODS:
        ax.plot(test.index, forecasts[method].values, label=method, lw=1, alpha=0.8)
    unit = cfg.get("duration_column", "duration")
    ax.set_title(f"{name} — test-period actual vs. forecast ({unit})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_forecasts_plot.png", dpi=150)
    plt.close(fig)


def plot_mape_comparison(rows: list[dict]) -> None:
    """Cross-domain MAPE bar chart — MAPE, not MAE/RMSE, because
    build_series() leaves durations in each dataset's native unit
    (hours/seconds/minutes); MAPE is unit-independent so it's the one
    metric safe to compare across datasets without a conversion pass."""
    if not rows:
        return
    datasets = [r["dataset"] for r in rows]
    x = np.arange(len(datasets))
    width = 0.8 / len(METHODS)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(METHODS):
        values = [r[f"{method}_mape"] for r in rows]
        ax.bar(x + i * width, values, width, label=method)
    ax.set_xticks(x + width * (len(METHODS) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Baseline MAPE by dataset and method")
    ax.legend()
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "baselines_mape_comparison.png", dpi=150)
    plt.close(fig)


def run_baselines(name: str, cfg: dict) -> dict:
    series = build_series(cfg)
    train, test = train_test_split(series)

    forecasts = {
        "naive": naive_forecast(series).loc[test.index],
        "historical_mean": historical_mean_forecast(series).loc[test.index],
        "moving_average": moving_average_forecast(series, MA_WINDOW).loc[test.index],
    }
    ses_fc, ses_alpha = ses_forecast(train, test)
    forecasts["ses"] = ses_fc

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {}
    errors = {}
    for method, fc in forecasts.items():
        metrics[method], errors[method] = compute_metrics(test, fc)

    # -------- persist per-test-point forecasts/errors
    out_df = pd.DataFrame({"actual": test})
    for method in METHODS:
        out_df[f"{method}_forecast"] = forecasts[method]
        out_df[f"{method}_error"] = errors[method]
    out_df.to_csv(OUT_DIR / f"{name}_baselines_forecasts.csv")

    plot_forecasts(name, cfg, test, forecasts)

    # -------- per-dataset report
    lines = [f"BASELINE FORECASTS — {name}", "=" * 60]
    lines.append(f"n_train: {len(train)}   n_test: {len(test)}")
    lines.append(f"train/test split: TRAIN_FRACTION={TRAIN_FRACTION} (data_quality.py Gate 3 rule)")
    lines.append("protocol: walk-forward, one-step-ahead (no lookahead; see "
                  "spec/baseline-forecasting-models.md)")
    lines.append("")
    lines.append("[methods]")
    lines.append(f"  naive:            y_hat_t = y_(t-1)")
    lines.append(f"  historical_mean:  y_hat_t = mean(y_1..y_(t-1))  (expanding window)")
    lines.append(f"  moving_average:   y_hat_t = mean(y_(t-k)..y_(t-1)),  k={MA_WINDOW} (fixed, all datasets)")
    lines.append(f"  ses:              y_hat_t = alpha*y_(t-1) + (1-alpha)*y_hat_(t-1),  "
                 f"alpha={ses_alpha:.4f} (fit on train only)")
    lines.append("")
    lines.append("[metrics] (error = actual - forecast; MAPE excludes |actual| <= 1e-9)")
    for method in METHODS:
        m = metrics[method]
        lines.append(f"  {method}: MAE={m['mae']}  RMSE={m['rmse']}  MAPE={m['mape']}%  "
                     f"(excluded {m['mape_excluded_zero_actuals']} zero-actual points)")
    lines.append("")
    lines.append(f"[output] per-point forecasts written to "
                 f"{OUT_DIR}/{name}_baselines_forecasts.csv")
    lines.append(f"[output] actual-vs-forecast plot written to "
                 f"{OUT_DIR}/{name}_forecasts_plot.png")
    Path(OUT_DIR / f"{name}_baselines.txt").write_text("\n".join(lines))

    return {
        "dataset": name,
        "n_train": len(train),
        "n_test": len(test),
        "ses_alpha": round(ses_alpha, 4),
        **{f"{method}_{k}": v for method in METHODS for k, v in metrics[method].items()
           if k in ("mae", "rmse", "mape")},
    }


def write_cross_domain_summary(rows: list[dict], skipped: list[tuple[str, str]]) -> None:
    lines = ["BASELINES CROSS-DOMAIN SUMMARY", "=" * 60]
    lines.append(f"moving_average window k={MA_WINDOW} (fixed across all datasets, regardless of "
                 f"frequency — see spec/baseline-forecasting-models.md's design-decisions table)")
    lines.append("")
    if rows:
        cols = list(rows[0].keys())
        lines.append(" | ".join(cols))
        for row in rows:
            lines.append(" | ".join(str(row[c]) for c in cols))
    else:
        lines.append("(no qualifying datasets)")
    if skipped:
        lines.append("")
        lines.append("SKIPPED (failed quality gate):")
        for name, report_path in skipped:
            lines.append(f"  {name}: see {report_path}")
    lines.append("")
    for row in rows:
        lines.append(json.dumps(row))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR / "baselines_cross_domain_summary.txt").write_text("\n".join(lines))


def main(argv: list[str]) -> int:
    cfg_paths = argv[1:] if len(argv) > 1 else FINAL_FOUR

    rows, skipped = [], []
    for cfg_path in cfg_paths:
        passed, cfg, report_path = quality_gate_passed(cfg_path)
        name = cfg.get("name", Path(cfg_path).stem)
        if not passed:
            skipped.append((name, report_path))
            print(f"{name}: SKIPPED (fails quality gate, see {report_path})")
            continue
        row = run_baselines(name, cfg)
        rows.append(row)
        print(f"{name}: baselines written to {OUT_DIR}/{name}_baselines.txt")

    write_cross_domain_summary(rows, skipped)
    plot_mape_comparison(rows)
    print(f"\ncross-domain summary written to {OUT_DIR}/baselines_cross_domain_summary.txt")
    print(f"cross-domain MAPE plot written to {OUT_DIR}/baselines_mape_comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
