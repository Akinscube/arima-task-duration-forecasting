"""
Model evaluation (metrics & benchmarks) — see spec/model-evaluation.md and
spec/model-evaluation-parameters.md.

*** Diebold-Mariano — PROVISIONAL, PENDING SUPERVISOR CONFIRMATION. ***
CLAUDE.md flags DM as "NOT confirmed by supervisor — do not implement as
final methodology without confirmation." This stage uses it anyway, on
the student's explicit instruction, after a diagnostic (Ljung-Box on all
16 dataset/baseline loss-differential series) found real serial
autocorrelation in 5 pairs — exactly what DM's HAC variance estimator is
designed to correct for and a plain paired test (Wilcoxon, this stage's
original plan) cannot. Every report/output states this plainly. See
spec/model-evaluation.md's Status note.

Reads ONLY the already-persisted per-point forecasts from
reports/arima/<name>_arima_forecasts.csv and
reports/baselines/<name>_baselines_forecasts.csv — never refits or
regenerates a forecast. For each dataset with successful output in both:

  1. Verifies the two CSVs' test-window indices are identical.
  2. For each of the four baselines, computes the paired absolute-error
     differential d_t = |baseline_error_t| - |arima_error_t|.
  3. Runs Ljung-Box on d_t (autocorrelation diagnostic, all 16 pairs).
  4. Runs the DM test (OLS of d_t on a constant, HAC/Newey-West cov).
  5. Applies Holm-Bonferroni correction within each dataset's 4-test
     family.
  6. Classifies each pair: arima_wins / baseline_wins /
     no_significant_difference / no_test_possible.

Output goes to reports/evaluation/ — kept separate from reports/,
reports/eda/, reports/baselines/, reports/arima/.

Usage:
    python src/evaluate_model.py                # the four final datasets
    python src/evaluate_model.py <name> [...]    # explicit subset (bare
                                                  # dataset names, not
                                                  # config paths)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

DATASETS = [
    "synthetic_it_support_tickets",
    "call_center_emea_2021_2025",
    "generated_emergency_dept_stays",
    "generated_job_shop_manufacturing",
]
BASELINE_METHODS = ["naive", "historical_mean", "moving_average", "ses"]

# spec/model-evaluation-parameters.md
ALPHA = 0.05
MAPE_ZERO_EPS = 1e-9
LJUNG_BOX_LAGS = [1, 5, 7]
DM_LABEL = "Diebold-Mariano — PROVISIONAL, PENDING SUPERVISOR CONFIRMATION"

ARIMA_DIR = Path("reports/arima")
BASELINES_DIR = Path("reports/baselines")
OUT_DIR = Path("reports/evaluation")


def load_pair(name: str) -> tuple[bool, pd.DataFrame, pd.DataFrame, str]:
    """Read-only join of the two prior stages' already-persisted per-point
    CSVs — never refits or regenerates a forecast (spec constraint)."""
    arima_path = ARIMA_DIR / f"{name}_arima_forecasts.csv"
    base_path = BASELINES_DIR / f"{name}_baselines_forecasts.csv"
    missing = [str(p) for p in (arima_path, base_path) if not p.exists()]
    if missing:
        return False, None, None, f"missing source file(s): {', '.join(missing)}"

    arima_df = pd.read_csv(arima_path, index_col=0, parse_dates=True)
    base_df = pd.read_csv(base_path, index_col=0, parse_dates=True)
    if not arima_df.index.equals(base_df.index):
        return False, None, None, "test-window index mismatch between arima and baselines CSVs"
    return True, arima_df, base_df, ""


def compute_metrics(actual: pd.Series, error: pd.Series) -> dict:
    """Same convention as baselines.py/arima_model.py: MAE/RMSE/MAPE from
    an already-computed error series (error = actual - forecast)."""
    abs_error = error.abs()
    mae = float(abs_error.mean())
    rmse = float(np.sqrt((error ** 2).mean()))
    nonzero = actual.abs() > MAPE_ZERO_EPS
    mape = float((abs_error[nonzero] / actual[nonzero].abs()).mean() * 100) if nonzero.any() else float("nan")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4) if not np.isnan(mape) else "nan"}


def ljung_box(d: np.ndarray) -> dict:
    lb = acorr_ljungbox(d, lags=LJUNG_BOX_LAGS, return_df=True)
    pvalues = {lag: float(lb.loc[lag, "lb_pvalue"]) for lag in LJUNG_BOX_LAGS}
    return {"pvalues": pvalues, "autocorrelated": any(p < ALPHA for p in pvalues.values())}


def dm_test(d: np.ndarray) -> dict:
    """Diebold-Mariano via OLS-on-constant with HAC (Newey-West) cov —
    mathematically the standard DM statistic (mean loss differential over
    its own autocorrelation-robust standard error). L uses the Newey-West
    automatic bandwidth, recomputed from this pair's own n (spec design
    decision: not the classical h-1=0 rule, since the observed
    autocorrelation comes from baseline misspecification, not
    forecast-horizon overlap)."""
    if np.allclose(d, 0.0, atol=1e-9):
        return {"possible": False}
    n = len(d)
    maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    fit = sm.OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "possible": True,
        "mean_d": float(d.mean()),
        "stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "maxlags": maxlags,
    }


def classify(arima_mae: float, baseline_mae: float, dm_result: dict, p_holm: float) -> str:
    if not dm_result["possible"]:
        return "no_test_possible"
    if p_holm >= ALPHA:
        return "no_significant_difference"
    return "arima_wins" if arima_mae < baseline_mae else "baseline_wins"


def evaluate_dataset(name: str, arima_df: pd.DataFrame, base_df: pd.DataFrame) -> dict:
    n_test = len(arima_df)
    arima_abs_err = arima_df["arima_error"].abs()
    arima_metrics = compute_metrics(arima_df["actual"], arima_df["arima_error"])

    pair_results = {}
    dm_results = {}
    for method in BASELINE_METHODS:
        base_error = base_df[f"{method}_error"]
        base_metrics = compute_metrics(base_df["actual"], base_error)
        d = (base_error.abs() - arima_abs_err).values

        lb = ljung_box(d)
        dm = dm_test(d)
        dm_results[method] = dm
        pair_results[method] = {"metrics": base_metrics, "ljung_box": lb, "dm": dm}

    # -------- Holm-Bonferroni within this dataset's family (only pairs where DM was possible)
    possible_methods = [m for m in BASELINE_METHODS if dm_results[m]["possible"]]
    if possible_methods:
        raw_pvals = [dm_results[m]["p_value"] for m in possible_methods]
        _, p_holm, _, _ = multipletests(raw_pvals, alpha=ALPHA, method="holm")
        holm_map = dict(zip(possible_methods, p_holm))
    else:
        holm_map = {}

    # -------- identical-baseline note (e.g. job-shop's naive == ses, spec risk).
    # atol=1e-4 (not a tighter float-equality tolerance): SES's alpha is fit by
    # numerical optimization and converges to ~0.9999999 rather than exactly
    # 1.0, so naive/ses errors differ by ~1e-7 even when "identical" at the
    # 4-decimal precision MAE/RMSE/MAPE are reported at.
    identical_pairs = []
    for i, m1 in enumerate(BASELINE_METHODS):
        for m2 in BASELINE_METHODS[i + 1:]:
            if np.allclose(base_df[f"{m1}_error"].values, base_df[f"{m2}_error"].values, atol=1e-4):
                identical_pairs.append((m1, m2))

    verdicts = {}
    for method in BASELINE_METHODS:
        dm = dm_results[method]
        p_holm = holm_map.get(method)
        verdicts[method] = classify(arima_metrics["mae"], pair_results[method]["metrics"]["mae"], dm, p_holm if p_holm is not None else 1.0)

    return {
        "name": name,
        "n_test": n_test,
        "arima_metrics": arima_metrics,
        "pair_results": pair_results,
        "holm_map": holm_map,
        "identical_pairs": identical_pairs,
        "verdicts": verdicts,
    }


def write_report(result: dict) -> None:
    name = result["name"]
    lines = [f"MODEL EVALUATION — {name}", "=" * 60, f"*** {DM_LABEL} ***", ""]
    lines.append(f"n_test: {result['n_test']}")
    lines.append("data sources: reports/arima/%s_arima_forecasts.csv, "
                 "reports/baselines/%s_baselines_forecasts.csv (read-only, not refit)" % (name, name))
    lines.append("loss differential: d_t = |baseline_error_t| - |arima_error_t| (positive = ARIMA smaller)")
    lines.append("'beats' requires BOTH a lower MAE AND Holm-adjusted p < alpha=%.2f" % ALPHA)
    lines.append("Holm-Bonferroni correction applied within this dataset's 4-test family only")
    if result["identical_pairs"]:
        for m1, m2 in result["identical_pairs"]:
            lines.append(f"NOTE: '{m1}' and '{m2}' baselines are numerically identical on this dataset "
                         f"(see reports/baselines/{name}_baselines.txt) — their DM tests vs ARIMA are "
                         f"expected to produce identical statistics/p-values, not a bug.")
    lines.append("")

    am = result["arima_metrics"]
    lines.append(f"[1] METRICS (MAE / RMSE / MAPE)")
    lines.append(f"    {'arima':16s}: MAE={am['mae']}  RMSE={am['rmse']}  MAPE={am['mape']}%")
    for method in BASELINE_METHODS:
        bm = result["pair_results"][method]["metrics"]
        lines.append(f"    {method:16s}: MAE={bm['mae']}  RMSE={bm['rmse']}  MAPE={bm['mape']}%")
    lines.append("")

    lines.append("[2] LJUNG-BOX (autocorrelation diagnostic on d_t — motivates the HAC correction below)")
    for method in BASELINE_METHODS:
        lb = result["pair_results"][method]["ljung_box"]
        p_str = ", ".join(f"lag{lag}: p={lb['pvalues'][lag]:.4f}" for lag in LJUNG_BOX_LAGS)
        flag = "AUTOCORRELATED" if lb["autocorrelated"] else "clean"
        lines.append(f"    vs {method:16s}: [{p_str}]  -> {flag}")
    lines.append("")

    lines.append(f"[3] {DM_LABEL}")
    for method in BASELINE_METHODS:
        dm = result["pair_results"][method]["dm"]
        p_holm = result["holm_map"].get(method)
        if not dm["possible"]:
            lines.append(f"    vs {method:16s}: identical forecasts, no test possible")
            continue
        lines.append(f"    vs {method:16s}: mean(d)={dm['mean_d']:+.4f}  stat={dm['stat']:+.3f}  "
                     f"p_raw={dm['p_value']:.4f}  p_holm={p_holm:.4f}  HAC maxlags={dm['maxlags']}  "
                     f"-> {result['verdicts'][method]}")
    lines.append("")

    wins = sum(1 for v in result["verdicts"].values() if v == "arima_wins")
    losses = sum(1 for v in result["verdicts"].values() if v == "baseline_wins")
    ties = sum(1 for v in result["verdicts"].values() if v == "no_significant_difference")
    lines.append(f"[4] SUMMARY: arima_wins={wins}  baseline_wins={losses}  "
                 f"no_significant_difference={ties}  (of {len(BASELINE_METHODS)} baselines)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR / f"{name}_evaluation.txt").write_text("\n".join(lines))


def plot_heatmap(results: list[dict]) -> None:
    verdict_code = {"arima_wins": 1, "no_significant_difference": 0, "baseline_wins": -1, "no_test_possible": np.nan}
    names = [r["name"] for r in results]
    grid = np.array([[verdict_code[r["verdicts"][m]] for m in BASELINE_METHODS] for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.8 * len(names)))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("lightgray")
    masked = np.ma.masked_invalid(grid)
    ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    for i, r in enumerate(results):
        for j, method in enumerate(BASELINE_METHODS):
            p_holm = r["holm_map"].get(method)
            label = f"{p_holm:.3f}" if p_holm is not None else "NA"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)

    ax.set_xticks(range(len(BASELINE_METHODS)))
    ax.set_xticklabels(BASELINE_METHODS, rotation=20, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(f"ARIMA vs. baselines — win/loss/tie\n({DM_LABEL})", fontsize=9)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "evaluation_win_loss_heatmap.png", dpi=150)
    plt.close(fig)


def write_cross_domain_summary(results: list[dict], skipped: list[tuple[str, str]]) -> None:
    lines = ["MODEL EVALUATION CROSS-DOMAIN SUMMARY", "=" * 60, f"*** {DM_LABEL} ***", ""]
    rows = []
    for r in results:
        wins = sum(1 for v in r["verdicts"].values() if v == "arima_wins")
        losses = sum(1 for v in r["verdicts"].values() if v == "baseline_wins")
        ties = sum(1 for v in r["verdicts"].values() if v == "no_significant_difference")
        not_possible = sum(1 for v in r["verdicts"].values() if v == "no_test_possible")
        rows.append({"dataset": r["name"], "n_test": r["n_test"], "arima_wins": wins,
                     "baseline_wins": losses, "no_significant_difference": ties, "no_test_possible": not_possible})

    if rows:
        cols = list(rows[0].keys())
        lines.append(" | ".join(cols))
        for row in rows:
            lines.append(" | ".join(str(row[c]) for c in cols))
    else:
        lines.append("(no evaluated datasets)")
    if skipped:
        lines.append("")
        lines.append("SKIPPED:")
        for name, reason in skipped:
            lines.append(f"  {name}: {reason}")
    lines.append("")
    for row in rows:
        lines.append(json.dumps(row))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR / "evaluation_cross_domain_summary.txt").write_text("\n".join(lines))


def main(argv: list[str]) -> int:
    names = argv[1:] if len(argv) > 1 else DATASETS

    results, skipped = [], []
    for name in names:
        ok, arima_df, base_df, reason = load_pair(name)
        if not ok:
            skipped.append((name, reason))
            print(f"{name}: SKIPPED ({reason})")
            continue
        result = evaluate_dataset(name, arima_df, base_df)
        write_report(result)
        results.append(result)
        print(f"{name}: evaluation written to {OUT_DIR}/{name}_evaluation.txt")

    write_cross_domain_summary(results, skipped)
    if results:
        plot_heatmap(results)
        print(f"win/loss heatmap written to {OUT_DIR}/evaluation_win_loss_heatmap.png")
    print(f"\ncross-domain summary written to {OUT_DIR}/evaluation_cross_domain_summary.txt")
    print(f"\n*** {DM_LABEL} ***")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
