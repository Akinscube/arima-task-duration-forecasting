"""
Analyse and interpret results — see spec/results-analysis.md and
spec/results-analysis-parameters.md.

Reads ONLY the trailing JSON summary lines already written by the four
prior stages (eda.py, baselines.py, arima_model.py, evaluate_model.py) —
no new statistic is computed here, this is a join. Merges them on
`dataset` into one master evidence table, one row per dataset that
appears in all four sources.

The actual RQ1/RQ2 interpretation is the separate, hand-written
reports/analysis/rq1_rq2_analysis.md — a script cannot produce that
narrative, only the reproducible table it cites.

Usage:
    python src/analyse_results.py
"""

import json
import sys
from pathlib import Path

SOURCES = {
    "eda": {
        "path": Path("reports/eda/eda_cross_domain_summary.txt"),
        "keys": ["dataset", "n", "frequency", "adf_p_d0", "adf_p_d1", "suggested_d",
                 "dominant_acf_lag", "dominant_pacf_lag", "cv", "seasonal_strength",
                 "visible_seasonality", "has_ground_truth"],
        "rename": {},
    },
    "baselines": {
        "path": Path("reports/baselines/baselines_cross_domain_summary.txt"),
        "keys": ["dataset", "n_train", "n_test", "ses_alpha", "naive_mae", "naive_rmse",
                 "naive_mape", "historical_mean_mae", "historical_mean_rmse", "historical_mean_mape",
                 "moving_average_mae", "moving_average_rmse", "moving_average_mape",
                 "ses_mae", "ses_rmse", "ses_mape"],
        "rename": {"n_train": "baselines_n_train", "n_test": "baselines_n_test"},
    },
    "arima": {
        "path": Path("reports/arima/arima_cross_domain_summary.txt"),
        "keys": ["dataset", "failed", "n_train", "n_test", "d", "p_max", "q_max",
                 "order", "aic_bic_agree", "mae", "rmse", "mape"],
        "rename": {"n_train": "arima_n_train", "n_test": "arima_n_test",
                   "mae": "arima_mae", "rmse": "arima_rmse", "mape": "arima_mape"},
    },
    "evaluation": {
        "path": Path("reports/evaluation/evaluation_cross_domain_summary.txt"),
        "keys": ["dataset", "n_test", "arima_wins", "baseline_wins",
                 "no_significant_difference", "no_test_possible"],
        "rename": {"n_test": "evaluation_n_test"},
    },
}

OUT_DIR = Path("reports/analysis")


def load_json_rows(source_name: str) -> dict[str, dict]:
    """Parse the trailing json.dumps(row) lines from a cross-domain
    summary file — a line is a data row iff it parses as a dict with a
    'dataset' key (skips the human-readable header/table above it).
    Fails loudly on any missing expected key, per spec: a master table
    that silently goes stale on a format drift is worse than a crash."""
    spec = SOURCES[source_name]
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(f"{source_name}: expected source file not found: {path}")

    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "dataset" not in row:
            continue

        missing = [k for k in spec["keys"] if k not in row]
        if missing:
            raise KeyError(f"{source_name} ({path}): row for {row.get('dataset')!r} "
                           f"is missing expected key(s) {missing}")

        renamed = {spec["rename"].get(k, k): row[k] for k in spec["keys"]}
        rows[row["dataset"]] = renamed
    return rows


def build_master_table() -> tuple[list[dict], list[str]]:
    """Inner join on `dataset` across all four sources — a dataset only
    gets a master-table row if it succeeded through all four stages."""
    per_source = {name: load_json_rows(name) for name in SOURCES}

    dataset_sets = [set(rows.keys()) for rows in per_source.values()]
    common = set.intersection(*dataset_sets)
    excluded = set.union(*dataset_sets) - common

    merged_rows = []
    for dataset in sorted(common):
        merged = {}
        for name in SOURCES:
            row = per_source[name][dataset]
            for k, v in row.items():
                if k == "dataset":
                    continue
                merged[k] = v
        merged_rows.append({"dataset": dataset, **merged})

    reasons = []
    for name, rows in per_source.items():
        for dataset in excluded:
            if dataset not in rows:
                reasons.append(f"{dataset}: missing from {name} cross-domain summary")

    return merged_rows, reasons


def write_outputs(rows: list[dict], reasons: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not rows:
        Path(OUT_DIR / "master_evidence_table.csv").write_text("")
        Path(OUT_DIR / "master_evidence_table.txt").write_text(
            "MASTER EVIDENCE TABLE\n" + "=" * 60 + "\n(no dataset present in all four sources)\n")
        return

    cols = list(rows[0].keys())

    import csv
    with open(OUT_DIR / "master_evidence_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["MASTER EVIDENCE TABLE", "=" * 60,
             "Joined from eda/baselines/arima/evaluation cross-domain summaries "
             "(no new computation — see spec/results-analysis.md).", ""]
    lines.append(" | ".join(cols))
    for row in rows:
        lines.append(" | ".join(str(row[c]) for c in cols))
    if reasons:
        lines.append("")
        lines.append("EXCLUDED (not present in all four sources):")
        for r in reasons:
            lines.append(f"  {r}")
    Path(OUT_DIR / "master_evidence_table.txt").write_text("\n".join(lines))


def main() -> int:
    rows, reasons = build_master_table()
    write_outputs(rows, reasons)
    print(f"master evidence table: {len(rows)} dataset(s) written to "
          f"{OUT_DIR}/master_evidence_table.{{csv,txt}}")
    for r in reasons:
        print(f"  EXCLUDED: {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
