"""
Cross-domain exploratory data analysis (EDA) — see spec/cross-domain-eda.md
and spec/eda-parameters.md.

Runs only against datasets that currently pass src/data_quality.py's
quality gate (re-checked at run time, not assumed from a past run). For
each qualifying dataset, produces:
  - basic EDA: descriptive stats, duration histogram, time series +
    rolling mean, seasonal boxplot
  - diagnostic EDA: ADF at d=0/d=1, ACF/PACF, STL decomposition
written to reports/eda/<name>_eda.txt plus plots — kept out of reports/
itself, which holds only Stage-0 quality reports and the recovery-
validation report.

Also writes reports/eda/eda_cross_domain_summary.txt, tabulating every
qualifying dataset on the same metrics (spec/eda-parameters.md's metric
list) as the evidence base for RQ2.

Usage:
    python src/eda.py                                # the four final datasets
    python src/eda.py configs/<name>.yaml [...]       # explicit subset
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.tsa.seasonal import STL

import data_quality  # reuse the canonical quality gate, not reimplement it

FINAL_FOUR = [
    "configs/synthetic_it_support_tickets.yaml",
    "configs/call_center_emea_2021_2025.yaml",
    "configs/generated_emergency_dept_stays.yaml",
    "configs/generated_job_shop_manufacturing.yaml",
]

# spec/eda-parameters.md
UNIT_TO_MINUTES = {
    "synthetic_it_support_tickets": 60.0,       # hours -> minutes
    "call_center_emea_2021_2025": 1.0 / 60.0,   # seconds -> minutes
    "generated_emergency_dept_stays": 1.0,      # already minutes
    "generated_job_shop_manufacturing": 1.0,    # already minutes
}
NATIVE_UNIT = {
    "synthetic_it_support_tickets": "hours",
    "call_center_emea_2021_2025": "seconds",
    "generated_emergency_dept_stays": "minutes",
    "generated_job_shop_manufacturing": "minutes",
}
ADF_ALPHA = 0.05
SEASONALITY_STRENGTH_THRESHOLD = 0.3

OUT_DIR = Path("reports/eda")
GROUND_TRUTH_PATH = Path("ground_truth.json")


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


def build_series(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw records + construct the aggregated series exactly as
    data_quality.py does (same drop rules, same resample/aggregate) —
    duplicated rather than imported, per cross-domain-eda.md's design
    decision, matching recover_structure.py's existing precedent."""
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
    return df, series


def describe(x: pd.Series) -> dict:
    x = x.dropna()
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sd": float(x.std()),
        "cv": float(x.std() / x.mean()) if x.mean() else float("nan"),
        "p25": float(x.quantile(0.25)),
        "p50": float(x.quantile(0.50)),
        "p75": float(x.quantile(0.75)),
        "p99": float(x.quantile(0.99)),
        "skew": float(skew(x)),
    }


def dominant_lag(x: pd.Series, nlags: int, func) -> tuple[int, float]:
    vals = func(x, nlags=nlags)
    tail = vals[1:]
    idx = int(np.argmax(np.abs(tail))) + 1
    return idx, float(tail[idx - 1])


def seasonal_strength(stl_result) -> float:
    var_resid = np.var(stl_result.resid)
    var_seasonal_resid = np.var(stl_result.seasonal + stl_result.resid)
    if var_seasonal_resid == 0:
        return 0.0
    return max(0.0, 1.0 - var_resid / var_seasonal_resid)


def run_eda(name: str, cfg: dict, has_ground_truth: bool) -> dict:
    df, series = build_series(cfg)
    unit_factor = UNIT_TO_MINUTES[name]
    native_unit = NATIVE_UNIT[name]
    freq = cfg.get("frequency", "D")
    seasonal_period = int(cfg.get("seasonal_period", 7))

    record_minutes = df["duration"] * unit_factor
    series_minutes = series * unit_factor

    record_stats = describe(record_minutes)
    series_stats = describe(series_minutes)

    # -------- ADF at d=0 and d=1 (fixed depth, spec/eda-parameters.md)
    s_clean = series.dropna()
    p_d0 = float(adfuller(s_clean)[1])
    diff_clean = series.diff().dropna()
    p_d1 = float(adfuller(diff_clean)[1])
    if p_d0 <= ADF_ALPHA:
        suggested_d = "0"
    elif p_d1 <= ADF_ALPHA:
        suggested_d = "1"
    else:
        suggested_d = "d>=2 (not resolved)"

    # -------- ACF/PACF
    nlags = min(40, len(s_clean) // 2 - 1)
    acf_lag, acf_val = dominant_lag(s_clean, nlags, lambda x, nlags: acf(x, nlags=nlags))
    pacf_lag, pacf_val = dominant_lag(s_clean, nlags, lambda x, nlags: pacf(x, nlags=nlags))

    # -------- STL decomposition (interpolated input only, per eda-parameters.md)
    stl_input = series.interpolate()
    stl_result = STL(stl_input, period=seasonal_period, robust=True).fit()
    strength = seasonal_strength(stl_result)
    visible_seasonality = "y" if strength > SEASONALITY_STRENGTH_THRESHOLD else "n"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------- Plot 1: histogram of record-level durations
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(record_minutes, bins=50)
    ax.set_title(f"{name} — record-level duration distribution (minutes)")
    ax.set_xlabel("duration (minutes)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_hist.png", dpi=150)
    plt.close(fig)

    # -------- Plot 2: time series + rolling mean
    fig, ax = plt.subplots(figsize=(11, 4))
    series_minutes.plot(ax=ax, lw=0.7, label="series")
    series_minutes.rolling(seasonal_period * 2, min_periods=1).mean().plot(
        ax=ax, lw=2, label="rolling mean")
    ax.set_title(f"{name} — {cfg.get('aggregation', 'median')} duration per {freq} (minutes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_series.png", dpi=150)
    plt.close(fig)

    # -------- Plot 3: seasonal boxplot
    fig, ax = plt.subplots(figsize=(9, 4))
    if freq == "D":
        groups = series_minutes.index.day_name()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        group_label = "day-of-week"
    else:
        groups = series_minutes.index.month_name()
        order = ["January", "February", "March", "April", "May", "June", "July",
                 "August", "September", "October", "November", "December"]
        group_label = "month-of-year"
    box_df = pd.DataFrame({"value": series_minutes.values, "group": groups})
    box_data = [box_df.loc[box_df["group"] == g, "value"].dropna() for g in order
                if (box_df["group"] == g).any()]
    box_labels = [g for g in order if (box_df["group"] == g).any()]
    ax.boxplot(box_data, tick_labels=box_labels)
    ax.set_title(f"{name} — seasonal subseries by {group_label} (minutes)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_seasonal_boxplot.png", dpi=150)
    plt.close(fig)

    # -------- Plot 4: ACF/PACF (raw + first-differenced)
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    plot_acf(s_clean, ax=axes[0, 0], lags=nlags, title="ACF (raw)")
    plot_pacf(s_clean, ax=axes[0, 1], lags=nlags, title="PACF (raw)", method="ywm")
    plot_acf(diff_clean, ax=axes[1, 0], lags=nlags, title="ACF (d=1)")
    plot_pacf(diff_clean, ax=axes[1, 1], lags=nlags, title="PACF (d=1)", method="ywm")
    fig.suptitle(f"{name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_acf_pacf.png", dpi=150)
    plt.close(fig)

    # -------- Plot 5: STL decomposition
    fig = stl_result.plot()
    fig.set_size_inches(10, 7)
    fig.suptitle(f"{name} — STL decomposition (period={seasonal_period})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_stl.png", dpi=150)
    plt.close(fig)

    # -------- missingness/outlier recap, sourced from the quality report
    quality_report = Path(f"reports/{name}_quality.txt").read_text().splitlines()
    integrity_lines = [l for l in quality_report if l.startswith("[integrity]")
                        or l.startswith("duration median")
                        or "missing periods" in l or "exact duplicate rows" in l]

    # -------- write the per-dataset report
    lines = [f"EDA REPORT — {name}", "=" * 60]
    lines.append(f"has_ground_truth: {'yes' if has_ground_truth else 'no'}")
    if not has_ground_truth:
        lines.append("NOTE: no ground truth available for this dataset — "
                      "all findings below are exploratory, not validated "
                      "against a known data-generating process.")
    lines.append(f"native unit: {native_unit}  (converted x{unit_factor:g} -> minutes below)")
    lines.append("")
    lines.append("[quality-gate recap, from reports/%s_quality.txt]" % name)
    lines.extend(f"  {l}" for l in integrity_lines)
    lines.append("")
    lines.append("[1] RECORD-LEVEL DESCRIPTIVE STATS (minutes)")
    lines.append(json.dumps(record_stats, indent=2))
    lines.append("")
    lines.append(f"[2] SERIES-LEVEL DESCRIPTIVE STATS (minutes, {freq} {cfg.get('aggregation','median')})")
    lines.append(json.dumps(series_stats, indent=2))
    lines.append("")
    lines.append("[3] STATIONARITY (ADF)")
    lines.append(f"    d=0: p={p_d0:.4f}  ({'stationary' if p_d0 <= ADF_ALPHA else 'non-stationary'} at alpha={ADF_ALPHA})")
    lines.append(f"    d=1: p={p_d1:.4f}  ({'stationary' if p_d1 <= ADF_ALPHA else 'non-stationary'} at alpha={ADF_ALPHA})")
    lines.append(f"    suggested_d: {suggested_d}")
    lines.append("")
    lines.append("[4] ACF / PACF")
    lines.append(f"    dominant ACF lag: {acf_lag} (value {acf_val:.3f})")
    lines.append(f"    dominant PACF lag: {pacf_lag} (value {pacf_val:.3f})")
    lines.append("")
    lines.append("[5] STL DECOMPOSITION")
    lines.append(f"    seasonal_strength (F_s): {strength:.3f}")
    lines.append(f"    visible_seasonality: {visible_seasonality}  (threshold {SEASONALITY_STRENGTH_THRESHOLD})")
    lines.append("    NOTE: STL input was linearly interpolated to fill any residual gaps;")
    lines.append("    ADF/ACF/PACF/descriptive stats above use the un-interpolated series.")
    lines.append("")
    lines.append(f"[6] PLOTS written to {OUT_DIR}/{name}_{{hist,series,seasonal_boxplot,acf_pacf,stl}}.png")
    Path(OUT_DIR / f"{name}_eda.txt").write_text("\n".join(lines))

    return {
        "dataset": name,
        "n": series_stats["n"],
        "frequency": freq,
        "adf_p_d0": round(p_d0, 4),
        "adf_p_d1": round(p_d1, 4),
        "suggested_d": suggested_d,
        "dominant_acf_lag": acf_lag,
        "dominant_pacf_lag": pacf_lag,
        "cv": round(series_stats["cv"], 3),
        "seasonal_strength": round(strength, 3),
        "visible_seasonality": visible_seasonality,
        "has_ground_truth": "y" if has_ground_truth else "n",
    }


def write_cross_domain_summary(rows: list[dict], skipped: list[tuple[str, str]]) -> None:
    lines = ["CROSS-DOMAIN EDA SUMMARY", "=" * 60]
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
    Path(OUT_DIR / "eda_cross_domain_summary.txt").write_text("\n".join(lines))


def main(argv: list[str]) -> int:
    cfg_paths = argv[1:] if len(argv) > 1 else FINAL_FOUR
    ground_truth_names = set()
    if GROUND_TRUTH_PATH.exists():
        ground_truth_names = set(json.loads(GROUND_TRUTH_PATH.read_text())["datasets"].keys())

    rows, skipped = [], []
    for cfg_path in cfg_paths:
        passed, cfg, report_path = quality_gate_passed(cfg_path)
        name = cfg.get("name", Path(cfg_path).stem)
        if not passed:
            skipped.append((name, report_path))
            print(f"{name}: SKIPPED (fails quality gate, see {report_path})")
            continue
        row = run_eda(name, cfg, has_ground_truth=name in ground_truth_names)
        rows.append(row)
        print(f"{name}: EDA written to {OUT_DIR}/{name}_eda.txt")

    write_cross_domain_summary(rows, skipped)
    print(f"\ncross-domain summary written to {OUT_DIR}/eda_cross_domain_summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
