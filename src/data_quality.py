"""
Dataset quality profiler — Stage 0/3 of the experimental pipeline.

Screens a candidate task-log dataset against the quality criteria defined in
the pipeline definition: task volume per period, calendar gaps, sufficient
history for ARIMA (training length, seasonal coverage, test length), and
basic integrity (duplicates, impossible durations, unit sanity).

Usage:
    python src/data_quality.py configs/<dataset>.yaml

Writes a plain-text report to reports/<dataset>_quality.txt and a series
plot to reports/<dataset>_series.png. Exit code 0 = all checks passed,
1 = one or more failures (so it can gate a screening loop).
"""

import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# ---------------------------------------------------------------- thresholds
MIN_TASKS_PER_PERIOD = 5      # aggregate below this is noise
MAX_MISSING_PCT = 0.05        # non-structural missingness ceiling
MIN_TRAIN_OBS = 200           # training observations (daily series target)
MIN_TEST_OBS = 30             # forecast points needed for stable metrics + DM test
MIN_SEASONAL_CYCLES = 3       # complete cycles required in training portion
TRAIN_FRACTION = 0.8


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["file"])
    ts_col = cfg["timestamp_column"]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    if "duration_column" in cfg:
        df["duration"] = pd.to_numeric(df[cfg["duration_column"]], errors="coerce")
    else:  # derive from start/end
        end = pd.to_datetime(df[cfg["end_column"]], errors="coerce")
        df["duration"] = (end - df[ts_col]).dt.total_seconds() / 60.0  # minutes

    return df


def profile(cfg: dict) -> tuple[list[str], dict]:
    lines, results = [], {}
    df = load_data(cfg)
    ts = cfg["timestamp_column"]
    freq = cfg.get("frequency", "D")
    agg = cfg.get("aggregation", "median")
    seasonal_period = int(cfg.get("seasonal_period", 7))
    name = cfg.get("name", Path(cfg["file"]).stem)

    lines.append(f"DATA QUALITY REPORT — {name}")
    lines.append(f"source file: {cfg['file']}")
    lines.append("=" * 60)

    # -------- Step 1: raw log profile
    n_raw = len(df)
    n_bad_ts = df[ts].isna().sum()
    df = df.dropna(subset=[ts])
    lines.append(f"\n[1] RAW LOG")
    lines.append(f"records: {n_raw}  (unparseable timestamps dropped: {n_bad_ts})")
    lines.append(f"date range: {df[ts].min()}  ->  {df[ts].max()}")

    dupes = df.duplicated().sum()
    lines.append(f"exact duplicate rows: {dupes}" + ("  <- review" if dupes else ""))

    # duration integrity
    n_nan_dur = df["duration"].isna().sum()
    n_neg = (df["duration"] < 0).sum()
    n_zero = (df["duration"] == 0).sum()
    p99 = df["duration"].quantile(0.99)
    dmax = df["duration"].max()
    lines.append(f"\n[integrity] duration NaN: {n_nan_dur}, negative: {n_neg}, zero: {n_zero}")
    lines.append(f"duration median: {df['duration'].median():.1f}, p99: {p99:.1f}, max: {dmax:.1f}")
    if dmax > 20 * p99:
        lines.append("  <- max is >>p99: check for open-ticket artefacts or mixed units")
    df = df[(df["duration"] > 0)].dropna(subset=["duration"])

    # tasks per period (irregularity)
    counts = df.set_index(ts).resample(freq).size()
    thin = (counts[counts > 0] < MIN_TASKS_PER_PERIOD).mean()
    lines.append(f"\n[volume] tasks per {freq}-period: mean {counts.mean():.1f}, "
                 f"min {counts.min()}, max {counts.max()}")
    lines.append(f"active periods below {MIN_TASKS_PER_PERIOD} tasks: {thin:.1%}")
    results["thin_periods_ok"] = thin < 0.10
    if not results["thin_periods_ok"]:
        lines.append(f"  FAIL: >10% of periods too thin — consider coarser frequency (e.g. W)")

    # -------- Step 2: series construction + gaps
    series = df.set_index(ts)["duration"].resample(freq).agg(agg)
    series_full = series.asfreq(freq)
    missing_pct = series_full.isna().mean()
    gap_groups = series_full.isna().astype(int).groupby(series_full.notna().cumsum()).sum()
    longest_gap = int(gap_groups.max()) if len(gap_groups) else 0

    lines.append(f"\n[2] SERIES ({agg} duration per {freq})")
    lines.append(f"length on regular calendar: {len(series_full)}")
    lines.append(f"missing periods: {missing_pct:.1%}   longest gap: {longest_gap} periods")
    lines.append("NOTE: classify gaps before judging — structural closure (e.g. weekends)")
    lines.append("      is not data loss; if structural, model business days or use W frequency.")
    results["gaps_ok"] = (missing_pct <= MAX_MISSING_PCT) and (longest_gap < seasonal_period)
    if not results["gaps_ok"]:
        lines.append(f"  FAIL: missing > {MAX_MISSING_PCT:.0%} or gap >= one seasonal period")

    # -------- Step 3: sufficient history
    n = series_full.notna().sum()
    n_train = int(n * TRAIN_FRACTION)
    n_test = n - n_train
    cycles_in_train = n_train / seasonal_period
    lines.append(f"\n[3] HISTORY (train fraction {TRAIN_FRACTION:.0%})")
    lines.append(f"usable observations: {n}  ->  train {n_train} / test {n_test}")
    lines.append(f"seasonal period {seasonal_period}: {cycles_in_train:.1f} cycles in training")
    results["train_ok"] = n_train >= MIN_TRAIN_OBS
    results["test_ok"] = n_test >= MIN_TEST_OBS
    results["seasonal_ok"] = cycles_in_train >= MIN_SEASONAL_CYCLES
    if not results["train_ok"]:
        lines.append(f"  FAIL: train < {MIN_TRAIN_OBS} observations")
    if not results["test_ok"]:
        lines.append(f"  FAIL: test < {MIN_TEST_OBS} forecast points")
    if not results["seasonal_ok"]:
        lines.append(f"  FAIL: < {MIN_SEASONAL_CYCLES} seasonal cycles in training")

    # -------- Step 4: eyeball plot (structural breaks)
    Path("reports").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    series_full.plot(ax=ax, lw=0.7, label="series")
    series_full.rolling(seasonal_period * 2, min_periods=1).mean().plot(
        ax=ax, lw=2, label="rolling mean")
    ax.set_title(f"{name} — {agg} duration per {freq}")
    ax.legend()
    fig.tight_layout()
    plot_path = f"reports/{name}_series.png"
    fig.savefig(plot_path, dpi=150)
    lines.append(f"\n[4] PLOT saved to {plot_path}")
    lines.append("Inspect for level shifts / structural breaks — a mid-series process")
    lines.append("change is a finding for the cross-domain analysis, record it.")

    # -------- verdict
    lines.append("\n" + "=" * 60)
    passed = all(results.values())
    results = {k: bool(v) for k, v in results.items()}
    lines.append("CHECKS: " + json.dumps(results))
    lines.append("VERDICT: " + ("PASS — candidate acceptable"
                                if passed else "FAIL — see items above"))
    return lines, {"passed": passed, "name": name}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    cfg = load_config(sys.argv[1])
    lines, outcome = profile(cfg)
    report = "\n".join(lines)
    Path("reports").mkdir(exist_ok=True)
    out = f"reports/{outcome['name']}_quality.txt"
    Path(out).write_text(report)
    print(report)
    print(f"\nreport written to {out}")
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())