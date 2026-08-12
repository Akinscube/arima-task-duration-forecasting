"""
Recovery-validation stage — checks whether the ARIMA identification pipeline
recovers the KNOWN planted structure of each researcher-generated dataset
(ground_truth.json, schema documented in
spec/ground-truth-schema.md).

Purpose: before trusting the pipeline on datasets whose
structure is unknown, demonstrate that the same pipeline recovers structure
that is known by construction.

For each dataset it:
  1. Builds the daily-median duration series exactly as data_quality.py does
     (same resample/aggregation), so screening and validation see one series.
  2. Runs ADF on the raw series (borderline results are expected and
     recorded when the ground truth's `seasonal_planted` flag is true) and
     on the deterministic residual (expected stationary).
  3. Recovers deterministic components by OLS on [weekend dummy, annual
     cosine] where the ground truth plants them, using the planted
     `annual_phase_reference_day` for the cosine phase, and compares
     estimated against planted effect sizes.
  4. Fits AR(1) or ARMA(1,1) to the residual per the ground truth's
     `stochastic_form` tag and compares recovered coefficients against
     planted values, reporting attenuation explicitly.
  5. Checks each recovered value against an explicit, per-dataset
     acceptance tolerance (TOLERANCES below) and reports PASS/FAIL.

Usage:
    python src/recover_structure.py ground_truth.json

Writes reports/recovery_validation.txt and prints the planted-vs-recovered
table.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Acceptance tolerances (T5, spec/synthetic-recovery-validation.md).
# Declared here, not in ground truth: ground truth is the planted DGP,
# tolerance is a validator-stage judgment call about "how close is close
# enough" and must be stated explicitly rather than inferred at run time.
TOLERANCES = {
    "generated_emergency_dept_stays": {
        "weekend_effect_minutes": 3.0,
        "annual_amplitude_minutes": 3.0,
        "ar1_phi_must_attenuate": True,
        "adf_resid_p_max": 0.05,
    },
    "generated_job_shop_manufacturing": {
        "arma_phi": 0.05,
        "arma_theta": 0.08,
        "adf_resid_p_max": 0.05,
        "aic_prefers_planted_form": True,
    },
}


def build_series(ds_cfg: dict, data_dir: str = "data/raw") -> pd.Series:
    """Same construction as data_quality.py: median duration per day."""
    name = ds_cfg["_name"]
    df = pd.read_csv(f"{data_dir}/{name}.csv",
                     parse_dates=[ds_cfg["timestamp_column"]])
    dur = ds_cfg["duration_column"]
    df = df.dropna(subset=[dur])
    df = df[df[dur] > 0]
    s = (df.set_index(ds_cfg["timestamp_column"])[dur]
           .resample("D").median().asfreq("D"))
    return s


def deterministic_design(idx: pd.DatetimeIndex, planted: dict, ds_name: str):
    """Build OLS design matrix for the components the ground truth plants."""
    cols, names = [np.ones(len(idx))], ["const"]
    if "weekend_effect_minutes" in planted:
        cols.append(np.isin(idx.dayofweek, [5, 6]).astype(float))
        names.append("weekend")
    if "annual_amplitude_minutes" in planted:
        if "annual_phase_reference_day" not in planted:
            raise ValueError(
                f"{ds_name}: latent_series_dgp plants annual_amplitude_minutes "
                f"but is missing required annual_phase_reference_day")
        doy = idx.dayofyear.values
        phase_day = planted["annual_phase_reference_day"]
        cols.append(np.cos(2 * np.pi * (doy - phase_day) / 365.25))
        names.append("annual_cos")
    return np.column_stack(cols), names


def stochastic_form(planted: dict, ds_name: str):
    """Read the explicit stochastic_form tag; fail loudly if absent/unrecognized."""
    form = planted.get("stochastic_form")
    if form == "AR1":
        return planted["ar1_phi"], None
    if form == "ARMA11":
        return planted["arma_phi"], planted["arma_theta"]
    raise ValueError(
        f"{ds_name}: latent_series_dgp has missing/unrecognized "
        f"stochastic_form ({form!r}); expected 'AR1' or 'ARMA11'")


def validate(ds_name: str, ds_cfg: dict, lines: list):
    A = lines.append
    planted = ds_cfg["latent_series_dgp"]
    ds_cfg["_name"] = ds_name
    s = build_series(ds_cfg)

    if "seasonal_planted" not in planted:
        raise ValueError(
            f"{ds_name}: latent_series_dgp missing required seasonal_planted flag")

    A(f"\n--- {ds_name} " + "-" * max(0, 50 - len(ds_name)))
    A(f"series: {len(s)} daily observations, "
      f"{s.isna().sum()} missing")

    # [1] ADF on raw series
    p_raw = adfuller(s.dropna())[1]
    A(f"[ADF] raw series p = {p_raw:.4f}"
      + ("   (deterministic seasonal/annual components planted -> "
         "borderline expected)" if planted["seasonal_planted"] else ""))

    # [2] deterministic recovery by OLS
    X, names = deterministic_design(s.index, planted, ds_name)
    y = s.values
    mask = ~np.isnan(y)
    ols = sm.OLS(y[mask], X[mask]).fit()
    resid = pd.Series(np.full(len(s), np.nan), index=s.index)
    resid[mask] = ols.resid

    A("[OLS] deterministic components (planted -> recovered):")
    est = dict(zip(names, ols.params))
    A(f"    base level: {planted['base_level_minutes']:.1f} -> "
      f"{est['const']:.1f}")
    if "weekend" in est:
        A(f"    weekend effect: {planted['weekend_effect_minutes']:.1f} -> "
          f"{est['weekend']:.2f}")
    if "annual_cos" in est:
        A(f"    annual amplitude: {planted['annual_amplitude_minutes']:.1f} -> "
          f"{est['annual_cos']:.2f}")

    p_resid = adfuller(resid.dropna())[1]
    A(f"[ADF] deterministic residual p = {p_resid:.2e}  (expected stationary)")

    # [3] stochastic recovery: AR(1) and ARMA(1,1) on residual
    phi, theta = stochastic_form(planted, ds_name)
    r = resid.dropna()

    ar1 = ARIMA(r, order=(1, 0, 0)).fit()
    arma = ARIMA(r, order=(1, 0, 1)).fit()
    A("[ARIMA] stochastic component (planted -> recovered):")
    A(f"    AR(1):     phi {phi:.2f} -> {ar1.params['ar.L1']:.3f}"
      f"   [AIC {ar1.aic:.1f}]")
    line = (f"    ARMA(1,1): phi {phi:.2f} -> {arma.params['ar.L1']:.3f}, "
            f"theta {theta if theta is not None else 0:.2f} -> "
            f"{arma.params['ma.L1']:.3f}   [AIC {arma.aic:.1f}]")
    A(line)

    atten = phi - ar1.params["ar.L1"]
    if atten > 0.05:
        A(f"    NOTE: AR(1) phi attenuated by {atten:.2f} — aggregation-"
          f"induced noise (record-level noise + daily-median aggregation, "
          f"expected per ground_truth.json's expected_recovery_note)")
    best = "AR(1)" if ar1.aic < arma.aic else "ARMA(1,1)"
    A(f"    AIC-preferred order: {best}"
      + ("   (matches planted form)" if
         (best == "ARMA(1,1)") == (theta is not None) else
         "   <- does not match planted form: record and discuss"))

    # [4] acceptance tolerance check (T5)
    tol = TOLERANCES.get(ds_name, {})
    checks = []
    if "weekend" in est and "weekend_effect_minutes" in tol:
        ok = abs(est["weekend"] - planted["weekend_effect_minutes"]) <= tol["weekend_effect_minutes"]
        checks.append((f"weekend effect within +/-{tol['weekend_effect_minutes']:.1f} min", ok))
    if "annual_cos" in est and "annual_amplitude_minutes" in tol:
        ok = abs(est["annual_cos"] - planted["annual_amplitude_minutes"]) <= tol["annual_amplitude_minutes"]
        checks.append((f"annual amplitude within +/-{tol['annual_amplitude_minutes']:.1f} min", ok))
    if tol.get("ar1_phi_must_attenuate"):
        ok = ar1.params["ar.L1"] < phi
        checks.append(("AR(1) phi attenuated below planted (expected)", ok))
    if "arma_phi" in tol:
        ok = abs(arma.params["ar.L1"] - phi) <= tol["arma_phi"]
        checks.append((f"ARMA phi within +/-{tol['arma_phi']:.2f}", ok))
    if "arma_theta" in tol and theta is not None:
        ok = abs(arma.params["ma.L1"] - theta) <= tol["arma_theta"]
        checks.append((f"ARMA theta within +/-{tol['arma_theta']:.2f}", ok))
    if "adf_resid_p_max" in tol:
        ok = p_resid <= tol["adf_resid_p_max"]
        checks.append((f"deterministic residual stationary (p <= {tol['adf_resid_p_max']})", ok))
    if tol.get("aic_prefers_planted_form"):
        ok = (best == "ARMA(1,1)") == (theta is not None)
        checks.append(("AIC prefers planted stochastic form", ok))

    overall_pass = all(ok for _, ok in checks) if checks else None
    A("[CHECK] acceptance tolerance (T5):")
    for label, ok in checks:
        A(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    A(f"    overall: {'PASS' if overall_pass else 'FAIL' if overall_pass is False else 'NO TOLERANCE DEFINED'}")

    return {
        "dataset": ds_name,
        "planted_phi": phi, "recovered_phi_ar1": round(float(ar1.params["ar.L1"]), 3),
        "recovered_phi_arma": round(float(arma.params["ar.L1"]), 3),
        "planted_theta": theta,
        "recovered_theta": round(float(arma.params["ma.L1"]), 3),
        "adf_raw_p": round(float(p_raw), 4),
        "adf_resid_p": float(p_resid),
        "aic_preferred": best,
        "tolerance_pass": overall_pass,
        "failed_checks": [label for label, ok in checks if not ok],
    }


def main() -> int:
    gt_path = sys.argv[1] if len(sys.argv) > 1 else "ground_truth.json"
    gt = json.loads(Path(gt_path).read_text())

    lines = [f"RECOVERY VALIDATION — ground truth {gt['version']} "
             f"(seed {gt['seed']})", "=" * 60]
    summary = []
    for name, cfg in gt["datasets"].items():
        summary.append(validate(name, cfg, lines))

    lines.append("\n" + "=" * 60)
    lines.append("SUMMARY (planted -> recovered):")
    for row in summary:
        lines.append(json.dumps(row))
    report = "\n".join(lines)

    Path("reports").mkdir(exist_ok=True)
    out = "reports/recovery_validation.txt"
    Path(out).write_text(report)
    print(report)
    print(f"\nreport written to {out}")
    return 0 if all(r["tolerance_pass"] is not False for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
