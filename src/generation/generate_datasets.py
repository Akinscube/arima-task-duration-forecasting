"""
generate_datasets.py
Synthetic dataset generation for two operational domains:
  1. emergency_dept_stays      (healthcare / ED length-of-stay)
  2. job_shop_manufacturing    (manufacturing / job processing time)

Design contract (see spec/synthetic-recovery-validation.md and
spec/ground-truth-schema.md):
  - Record-level task logs are generated from an explicit, planted daily
    data-generating process (DGP) on the latent series level.
  - All DGP parameters are written to ground_truth.json BEFORE any
    modelling, so recovery validation (recover_structure.py) can be run
    against a pre-registered truth.
  - Dataset keys match the corresponding configs/<name>.yaml `name:` field
    exactly; CSVs are written to data/raw/<name>.csv.
  - Known caveat: per-record noise + median aggregation attenuates
    recovered AR coefficients (documented finding, Section 3.7).

Reproducibility: single fixed seed for the whole run; no system entropy.
"""

import json
import numpy as np
import pandas as pd

SEED = 30124470  # enrolment number -> fixed, citable seed
rng = np.random.default_rng(SEED)

DATES = pd.date_range("2022-01-01", "2025-12-31", freq="D")
N_DAYS = len(DATES)  # 1461

GROUND_TRUTH = {
    "version": "v1",
    "seed": SEED,
    "generated_for": "IS4T702 dissertation - cross-domain datasets",
    "date_range": ["2022-01-01", "2025-12-31"],
    "datasets": {},
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def ar1(n, phi, sigma, rng):
    """Stationary AR(1), zero mean."""
    x = np.zeros(n)
    x[0] = rng.normal(0, sigma / np.sqrt(1 - phi**2))
    e = rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def arma11(n, phi, theta, sigma, rng):
    """Stationary ARMA(1,1), zero mean."""
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t] + theta * e[t - 1]
    return x


# ======================================================================
# 1. EMERGENCY DEPARTMENT STAYS
#    Latent daily median LOS (minutes):
#      m_t = 180 + 25*I[weekend] + 20*cos(2*pi*(doy-15)/365.25)  (winter peak ~15 Jan)
#            + AR(1)(phi=0.60, sigma=5)
#    Record LOS_i = m_t * acuity_factor_i * exp(N(0, 0.35)), clipped to [10, 1440]
#    Dropout: LWBS (left without being seen) -> los_minutes = NaN
#      P(LWBS) = 0.04 base, 0.07 during winter months (Dec-Feb)
#    Volume: arrivals_t ~ Poisson(85 * 0.92^weekend * 1.18^winter)
# ======================================================================

def gen_emergency_dept():
    doy = DATES.dayofyear.values
    weekend = np.isin(DATES.dayofweek, [5, 6]).astype(float)
    winter = np.isin(DATES.month, [12, 1, 2]).astype(float)

    annual_phase_reference_day = 15
    annual = 20.0 * np.cos(2 * np.pi * (doy - annual_phase_reference_day) / 365.25)
    ar = ar1(N_DAYS, phi=0.60, sigma=5.0, rng=rng)
    m = 180.0 + 25.0 * weekend + annual + ar  # latent daily median LOS

    lam = 85.0 * np.where(weekend == 1, 0.92, 1.0) * np.where(winter == 1, 1.18, 1.0)
    arrivals = rng.poisson(lam)

    acuity_levels = np.array([1, 2, 3, 4, 5])
    acuity_probs = np.array([0.05, 0.15, 0.40, 0.30, 0.10])
    acuity_factor = {1: 2.20, 2: 1.60, 3: 1.00, 4: 0.75, 5: 0.55}
    complaint_cats = ["cardiac", "respiratory", "trauma", "abdominal",
                      "neuro", "msk", "infection", "other"]
    areas = ["majors", "minors", "resus", "paeds"]

    rows = []
    eid = 100000
    for t, (date, n_arr) in enumerate(zip(DATES, arrivals)):
        ac = rng.choice(acuity_levels, size=n_arr, p=acuity_probs)
        noise = rng.normal(0.0, 0.35, size=n_arr)
        base = m[t] * np.array([acuity_factor[a] for a in ac]) * np.exp(noise)
        los = np.clip(base, 10, 1440)

        p_lwbs = 0.07 if winter[t] == 1 else 0.04
        lwbs = rng.random(n_arr) < p_lwbs

        arr_min = np.sort(rng.integers(0, 24 * 60, size=n_arr))
        for i in range(n_arr):
            eid += 1
            arrival = date + pd.Timedelta(minutes=int(arr_min[i]))
            if lwbs[i]:
                dur, dep, dispo = np.nan, pd.NaT, "LWBS"
            else:
                dur = round(float(los[i]), 1)
                dep = arrival + pd.Timedelta(minutes=dur)
                dispo = rng.choice(["Discharged", "Admitted", "Transfer"],
                                   p=[0.70, 0.26, 0.04])
            rows.append((f"ED{eid}", arrival, dep, int(ac[i]),
                         rng.choice(complaint_cats), rng.choice(areas),
                         dispo, dur))

    df = pd.DataFrame(rows, columns=[
        "encounter_id", "arrival_ts", "departure_ts", "acuity_level",
        "chief_complaint_category", "attending_area", "disposition",
        "los_minutes"])
    df.to_csv("data/raw/generated_emergency_dept_stays.csv", index=False)

    GROUND_TRUTH["datasets"]["generated_emergency_dept_stays"] = {
        "domain": "healthcare_emergency_department",
        "duration_column": "los_minutes",
        "timestamp_column": "arrival_ts",
        "aggregation": "median per D",
        "latent_series_dgp": {
            "form": "m_t = 180 + 25*weekend + 20*cos(2pi*(doy-15)/365.25) + AR1",
            "base_level_minutes": 180.0,
            "weekend_effect_minutes": 25.0,
            "annual_amplitude_minutes": 20.0,
            "annual_peak": "mid-January (winter surge)",
            "annual_phase_reference_day": annual_phase_reference_day,
            "seasonal_planted": True,
            "stochastic_form": "AR1",
            "ar1_phi": 0.60,
            "ar1_sigma": 5.0,
        },
        "record_level": {
            "form": "los_i = m_t * acuity_factor * exp(N(0, 0.35)), clip [10, 1440]",
            "acuity_probs": {"1": 0.05, "2": 0.15, "3": 0.40, "4": 0.30, "5": 0.10},
            "acuity_factors": acuity_factor,
            "lognormal_sigma": 0.35,
        },
        "dropout": {
            "mechanism": "LWBS -> los_minutes NaN",
            "p_base": 0.04, "p_winter": 0.07,
        },
        "volume": {
            "form": "Poisson(85 * 0.92^weekend * 1.18^winter)",
            "base_lambda": 85, "weekend_mult": 0.92, "winter_mult": 1.18,
        },
        "expected_recovery_note": ("AR(1) phi expected to attenuate below 0.60 "
                                   "in the aggregated median series due to "
                                   "record-level noise (v2 finding, Sec 3.7)."),
    }
    return df


# ======================================================================
# 2. JOB-SHOP MANUFACTURING
#    Latent daily median processing time (minutes):
#      p_t = 55 + ARMA(1,1)(phi=0.85, theta=0.35, sigma=1.2)   [high persistence]
#    Record proc_i = p_t * exp(N(0, 0.08))   [tight engineered tolerances]
#    Volume: jobs_t ~ Poisson(45), 7-day continuous operation, no dropout
#    Rework flag on 2% of jobs (duration retained).
# ======================================================================

def gen_job_shop():
    x = arma11(N_DAYS, phi=0.85, theta=0.35, sigma=2.0, rng=rng)
    p = 55.0 + x

    jobs = rng.poisson(45.0, N_DAYS)
    machines = [f"M{k:02d}" for k in range(1, 13)]
    families = ["housing", "shaft", "bracket", "gear", "plate", "flange"]
    shifts = ["A", "B", "C"]

    rows = []
    jid = 500000
    for t, (date, n_jobs) in enumerate(zip(DATES, jobs)):
        noise = rng.normal(0.0, 0.04, size=n_jobs)
        proc = np.clip(p[t] * np.exp(noise), 20, 150)
        start_min = np.sort(rng.integers(0, 24 * 60, size=n_jobs))
        rework = rng.random(n_jobs) < 0.02
        for i in range(n_jobs):
            jid += 1
            start = date + pd.Timedelta(minutes=int(start_min[i]))
            dur = round(float(proc[i]), 1)
            rows.append((f"J{jid}", start,
                         start + pd.Timedelta(minutes=dur),
                         rng.choice(machines), rng.choice(families),
                         shifts[start.hour // 8], bool(rework[i]), dur))

    df = pd.DataFrame(rows, columns=[
        "job_id", "actual_start", "completion_ts", "machine_id",
        "part_family", "operator_shift", "rework_flag",
        "processing_minutes"])
    df.to_csv("data/raw/generated_job_shop_manufacturing.csv", index=False)

    GROUND_TRUTH["datasets"]["generated_job_shop_manufacturing"] = {
        "domain": "manufacturing_job_shop",
        "duration_column": "processing_minutes",
        "timestamp_column": "actual_start",
        "aggregation": "median per D",
        "latent_series_dgp": {
            "form": "p_t = 55 + ARMA(1,1)",
            "base_level_minutes": 55.0,
            "seasonal_planted": False,
            "stochastic_form": "ARMA11",
            "arma_phi": 0.85,
            "arma_theta": 0.35,
            "arma_sigma": 2.0,
            "seasonality": "none planted (persistence-dominated domain)",
        },
        "record_level": {
            "form": "proc_i = p_t * exp(N(0, 0.04)), clip [20, 150]",
            "lognormal_sigma": 0.04,
            "note": "tight engineered tolerances -> low aggregation noise, "
                    "ARMA(1,1) expected to recover faithfully (v2 finding)",
        },
        "dropout": {"mechanism": "none", "rework_flag_rate": 0.02},
        "volume": {"form": "Poisson(45)", "base_lambda": 45,
                   "operation": "7-day continuous"},
    }
    return df


if __name__ == "__main__":
    from pathlib import Path
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    ed = gen_emergency_dept()
    js = gen_job_shop()
    with open("ground_truth.json", "w") as f:
        json.dump(GROUND_TRUTH, f, indent=2, default=str)
    print(f"generated_emergency_dept_stays: {len(ed):,} records "
          f"({ed['los_minutes'].isna().sum():,} LWBS NaN)")
    print(f"generated_job_shop_manufacturing: {len(js):,} records")
    print("ground_truth.json written (pre-registered before modelling)")
