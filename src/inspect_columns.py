"""
Column inspector — pre-flight check before writing a Stage 0 config.

Helps pick timestamp_column / duration_column (or end_column) values for a
new configs/<dataset>.yaml by reporting, for a raw CSV: column overview
(dtype, null %, samples), timestamp-candidate parse rate + date range +
day/month ambiguity warning, and duration-candidate stats + unit-consistency
check against an end column if given.

Detection rules are specified in spec/column-inspection-rules.md — this
tool implements T1a/T1b (overview + ambiguity) and T2 (unit consistency).

Usage:
    python src/inspect_columns.py <csv_path>
    python src/inspect_columns.py <csv_path> --ts COL
    python src/inspect_columns.py <csv_path> --ts COL --duration COL
    python src/inspect_columns.py <csv_path> --ts COL --end COL [--duration COL]
"""

import argparse
import sys

import pandas as pd

TIMESTAMP_LIKE_SAMPLE_SIZE = 200
TIMESTAMP_LIKE_PARSE_RATE = 0.9
AMBIGUITY_SAMPLE_SIZE = 500
AMBIGUOUS_DATE_PATTERN = r"^(\d{1,2})[-/](\d{1,2})[-/]\d{4}"

# duration unit-consistency bands (spec T2)
SAME_UNIT_BAND = (0.9, 1.1)
SECONDS_BAND = (55, 65)
HOURS_BAND = (1 / 65, 1 / 55)
MATCH_TOLERANCE_MINUTES = 1.0


def overview(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}\ncolumns: {len(df.columns)}")
    print("-" * 60)
    for col in df.columns:
        s = df[col]
        null_pct = s.isna().mean()
        samples = s.dropna().head(3).tolist()
        flag = ""
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sample = s.dropna().head(TIMESTAMP_LIKE_SAMPLE_SIZE)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > TIMESTAMP_LIKE_PARSE_RATE:
                flag = "  <- looks like a timestamp"
        print(f"{col:<30} {str(s.dtype):<10} null:{null_pct:>6.1%}  {samples}{flag}")


def check_timestamp(df: pd.DataFrame, col: str) -> pd.Series:
    raw = df[col]
    parsed = pd.to_datetime(raw, errors="coerce")
    n_bad = parsed.isna().sum() - raw.isna().sum()
    print(f"\n[timestamp] {col}")
    print(f"nulls: {raw.isna().sum()}  unparseable (non-null but failed): {n_bad}")
    print(f"range: {parsed.min()}  ->  {parsed.max()}")

    sample = raw.dropna().astype(str).head(AMBIGUITY_SAMPLE_SIZE)
    ambiguous = sample.str.match(AMBIGUOUS_DATE_PATTERN)
    if ambiguous.any():
        parts = sample[ambiguous].str.extract(AMBIGUOUS_DATE_PATTERN)
        both_le12 = ((parts[0].astype(int) <= 12) & (parts[1].astype(int) <= 12)).mean()
        if both_le12 > 0:
            print(f"  WARNING: {both_le12:.0%} of sampled rows have day<=12 AND month<=12 "
                  f"— DD-MM vs MM-DD is ambiguous here. pd.to_datetime(..) with no "
                  f"dayfirst= arg may silently misparse some rows. Verify format and "
                  f"consider normalizing to ISO before use.")
    return parsed


def check_duration(df: pd.DataFrame, col: str) -> pd.Series:
    dur = pd.to_numeric(df[col], errors="coerce")
    print(f"\n[duration] {col}")
    print(f"nulls: {dur.isna().sum()}  negative: {(dur < 0).sum()}  zero: {(dur == 0).sum()}")
    print(dur.describe())
    return dur


def check_unit_consistency(given: pd.Series, derived_minutes: pd.Series) -> None:
    mask = given.notna() & derived_minutes.notna() & (derived_minutes != 0)
    ratio = given[mask].median() / derived_minutes[mask].median()
    print(f"\n[unit consistency] median(given) / median(derived-minutes) = {ratio:.3f}")

    if SAME_UNIT_BAND[0] <= ratio <= SAME_UNIT_BAND[1]:
        band, rescaled = "same unit (minutes)", given
    elif SECONDS_BAND[0] <= ratio <= SECONDS_BAND[1]:
        band, rescaled = "given is in seconds", given / 60
    elif HOURS_BAND[0] <= ratio <= HOURS_BAND[1]:
        band, rescaled = "given is in hours", given * 60
    else:
        print(f"  band: unrecognized scale (ratio {ratio:.3f} matches no known band)")
        print("  match rate: undefined (unrecognized scale — inspect manually)")
        return

    match_rate = (derived_minutes[mask] - rescaled[mask]).abs().le(MATCH_TOLERANCE_MINUTES).mean()
    print(f"  band: {band}")
    print(f"  match rate (within {MATCH_TOLERANCE_MINUTES:.0f} min of derived): {match_rate:.1%}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path")
    p.add_argument("--ts", help="candidate timestamp_column")
    p.add_argument("--duration", help="candidate duration_column")
    p.add_argument("--end", help="candidate end_column (duration derived as end - ts)")
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)

    if not args.ts:
        overview(df)
        return 0

    ts = check_timestamp(df, args.ts)

    if args.end:
        end = check_timestamp(df, args.end)
        derived = (end - ts).dt.total_seconds() / 60.0
        print(f"\n[derived duration] ({args.end} - {args.ts}) in minutes")
        print(derived.describe())
        if args.duration:
            given = check_duration(df, args.duration)
            check_unit_consistency(given, derived)
    elif args.duration:
        check_duration(df, args.duration)

    return 0


if __name__ == "__main__":
    sys.exit(main())
