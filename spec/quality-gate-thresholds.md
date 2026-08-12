# Quality Gate Thresholds (T4)

Companion to [stage0-dataset-screening.md](stage0-dataset-screening.md) —
exact formulas and constants for `src/data_quality.py`'s four gates plus
its integrity checks, written down before implementation.

Constants (global, no per-config override — see stage0-dataset-screening.md's
design-decisions table):

| Constant | Value | Used by |
|---|---|---|
| `MIN_TASKS_PER_PERIOD` | 5 | volume gate |
| `MAX_MISSING_PCT` | 0.05 | gap gate |
| `MIN_TRAIN_OBS` | 200 | history gate (train) |
| `MIN_TEST_OBS` | 30 | history gate (test) |
| `MIN_SEASONAL_CYCLES` | 3 | history gate (seasonal) |
| `TRAIN_FRACTION` | 0.8 | history gate (train/test split) |

## Gate 1 — volume per period

- `counts` = row count per resampled period (config `frequency`, default `D`).
- `thin = mean over active periods (count > 0) of I[count < MIN_TASKS_PER_PERIOD]`
- **Pass** if `thin < 0.10` (fewer than 10% of active periods are thin).

## Gate 2 — calendar gaps

- Build the duration series at the configured frequency/aggregation, then
  reindex to a full regular calendar (`asfreq`).
- `missing_pct` = fraction of the full calendar that is NaN.
- `longest_gap` = longest consecutive run of NaN periods.
- **Pass** if `missing_pct <= MAX_MISSING_PCT` **and**
  `longest_gap < seasonal_period` (config, default 7).
- Note in the report: a gap should be classified (structural closure,
  e.g. weekends, vs. true data loss) before judging severity — the check
  itself does not distinguish the two.

## Gate 3 — sufficient history

- `n` = count of non-null periods on the full regular calendar.
- `n_train = floor(n * TRAIN_FRACTION)`, `n_test = n - n_train`.
- `cycles_in_train = n_train / seasonal_period`.
- **Pass (train)** if `n_train >= MIN_TRAIN_OBS`.
- **Pass (test)** if `n_test >= MIN_TEST_OBS`.
- **Pass (seasonal)** if `cycles_in_train >= MIN_SEASONAL_CYCLES`.

## Integrity checks (reported, not gating)

- **Duplicates**: exact full-row duplicate count (`df.duplicated()`, no
  subset/key column — see stage0-dataset-screening.md's
  duplicate-definition decision). Reported for review, does not fail the
  dataset on its own.
- **Duration sanity**: count of NaN / negative / zero durations, reported;
  rows with NaN or non-positive duration are dropped before the gates run.
- **Outlier flag**: if `max(duration) > 20 * p99(duration)`, flag for
  manual review (open-ticket artefacts or mixed units) — informational,
  not gating.

## Overall verdict

- `passed = thin_periods_ok AND gaps_ok AND train_ok AND test_ok AND seasonal_ok`
  (the three integrity checks above never affect `passed`).
- Exit code: `0` if `passed`, else `1`.
