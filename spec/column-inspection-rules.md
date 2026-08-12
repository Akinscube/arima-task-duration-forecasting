# Column Inspection Rules (T1 + T2)

Companion to [stage0-dataset-screening.md](stage0-dataset-screening.md) —
precise detection rules for `src/inspect_columns.py`, written down before
implementation so the heuristics aren't reverse-engineered from code later.

## T1a — "looks like a timestamp" overview flag

For each object-dtype column, in overview mode:
- Sample: the first 200 non-null values of the column.
- Parse: `pd.to_datetime(sample, errors="coerce")` (permissive — see
  stage0-dataset-screening.md's parsing-strictness decision).
- Flag the column if `parsed.notna().mean() > 0.9`.

Non-object-dtype columns are never flagged (a numeric or bool column is
never a timestamp candidate here).

## T1b — day/month ambiguity warning

Given a candidate timestamp column (`--ts COL`):
- Sample: the first 500 non-null values, as strings.
- Structural match: rows matching `^\d{1,2}[-/]\d{1,2}[-/]\d{4}` (i.e.
  shaped like `D[D]-M[M]-YYYY` or `D[D]/M[M]/YYYY` — a plain 4-digit-year
  numeric date with no textual month).
- Among structurally-matching rows, extract the first two numeric groups
  and compute the fraction where **both** are `<= 12`.
- Warn if that fraction is `> 0` — i.e. if at least one sampled row's date
  cannot be disambiguated between `DD-MM` and `MM-DD` from its digits
  alone. A single such row is enough to warn: it means the format cannot
  be verified from the data, even if most rows are unambiguous.
- Report the warning as: share of ambiguous rows, and a recommendation to
  normalize to ISO-8601 or pass an explicit format before use.

Out of scope (documented, not a gap): two-digit years, non-numeric month
names, and other locale-specific ambiguities are not detected.

## T2 — duration unit-consistency check

Given `--end COL` (paired with `--ts` to derive a duration) and
`--duration COL` (a given duration column) together:

1. `derived_minutes = (end - ts).total_seconds() / 60`
2. `ratio = median(given) / median(derived_minutes)`, computed over rows
   where both are non-null and `derived_minutes` is non-zero.
3. Classify `ratio` into a band:

| Band | Condition | Meaning |
|---|---|---|
| same-unit | `0.9 <= ratio <= 1.1` | `given` is already in minutes |
| seconds | `55 <= ratio <= 65` | `given` is in seconds |
| hours | `1/65 <= ratio <= 1/55` | `given` is in hours |
| unrecognized | none of the above | report the raw ratio; do not guess a scale |

4. Rescale `given` into minutes per the matched band (divide by 60 for
   seconds, multiply by 60 for hours, unchanged for same-unit).
5. Match rate = fraction of non-null rows where
   `abs(derived_minutes - rescaled_given) <= 1` (1-minute tolerance, to
   absorb rounding).
6. If the band is unrecognized, report the raw ratio and `match rate:
   undefined (unrecognized scale — inspect manually)` rather than
   computing a match rate against a guessed scale.
