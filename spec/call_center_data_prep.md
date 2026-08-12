# Data Preparation — Call Center Operations Dataset

Source: `data/raw/call_center_operations_wfo_dataset.csv`
Final config: `configs/call_center_emea_2021_2025.yaml`
Final data: `data/processed/call_center_emea_2021_2025.csv`
Quality report: `reports/call_center_emea_2021_2025_quality.txt`

## 1. Initial column-mapping correction

The config file was found pre-populated with column names copied from an unrelated
dataset (`Scheduled_Start`, `Processing_Time` — manufacturing job columns that don't
exist in this file). Inspection of the actual raw CSV header showed this dataset has
a fundamentally different structure: it is **pre-aggregated per reporting bucket** —
one row per `(Date, Hour, Country, Region, Site, Queue_Name, Channel, ...)`
combination — rather than one row per individual call.

- **Rows:** 10,000
- **Cardinality:** 53 countries × 58 sites × 20 queues × 3 channels × 5 business units
- **Duration proxy:** `aHT_Seconds` (average handle time for that bucket, in seconds)
  — there is no per-call duration in the raw data, only bucket-level averages.

## 2. Date format normalization

Raw `Date` column is formatted `DD-MM-YYYY` (e.g. `29-09-2023`). This is ambiguous
for any day ≤ 12 (e.g. `05-06-2023` could mean 5 June or 6 May), and the pipeline's
loader (`pd.to_datetime(..., errors="coerce")`) does not pass `dayfirst=True`, so it
would silently misparse a subset of rows without raising an error.

- **Fix:** parsed with `pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')`
  and rewrote to ISO 8601 (`YYYY-MM-DD`) in all prepared output files, removing the
  ambiguity permanently rather than relying on a parsing flag at load time.

## 3. Segment-scope decision (attempt 1 — failed)

Tested whether the series should pool all segments or scope to a single queue. Found
that a single `Queue_Name + Site` pair is far too sparse (~30 rows spread over ~5
years). Selected the densest single queue, **`RETENTION_VOICE_T2`**, pooled across
sites/countries (1,019 rows, 801 distinct days, span 2021-01 to 2026-12).

- **Output:** `data/processed/call_center_retention_voice_t2.csv`
- **Config:** `configs/call_center_operations_wfo_dataset.yaml`
  (`timestamp_column: Date`, `duration_column: aHT_Seconds`, `frequency: W`,
  `seasonal_period: 52`)
- **Result:** `VERDICT: FAIL`
  - `thin_periods_ok: false` — 71.7% of weeks <5 rows, mean 3.3 rows/week
  - `gaps_ok: false` — 11.0% missing weeks, longest gap 7 weeks
  - `train_ok` / `test_ok` / `seasonal_ok`: all passed (220 train / 56 test / 4.2
    seasonal cycles)

## 4. Root-cause analysis of the two failures

- **`thin_periods` cause:** single-queue scope is structurally too narrow a slice of
  the 10,000-row dataset to reach 5 rows/week reliably.
- **`gaps` cause:** traced every one of the 27 zero-row weeks in the *entire raw
  dataset* (all 10,000 rows, every segment) and found **100% of them fall in 2026**.
  2026 has only 450 rows vs. ~1,800-2,000/year for 2021-2025, tapering sharply after
  March 2026 (128 → 102 → 65 → 15 rows/month). This is an incomplete trailing year
  in the source data, not a recurring structural gap — confirmed by checking that
  gap weeks are absent regardless of which segment (region/queue/channel) is
  inspected.

## 5. Segment-density comparison

Weekly row density was computed for three broader pooling candidates to find the
smallest scope that clears the volume threshold:

| Scope                       | Rows  | Mean rows/week | Weeks <5 rows |
|------------------------------|-------|-----------------|----------------|
| `Business_Unit = E-commerce` | 2,081 | 6.7             | 24.9%          |
| `Region = EMEA`               | 5,613 | 18.1            | 10.0%          |
| `Channel = Chat`              | 3,293 | 10.6            | 12.0%          |

`Region = EMEA` (the largest region, 56% of all rows) was selected as the best
density-vs-coherence tradeoff.

## 6. Final transformation applied

Filtered the raw CSV to:

```
Region == 'EMEA'  AND  Date <= '2025-12-31'
```

- Reformatted `Date` to ISO 8601.
- **Output:** `data/processed/call_center_emea_2021_2025.csv`
  (5,366 rows, 2021-01-01 → 2025-12-31)
- **Config:** `configs/call_center_emea_2021_2025.yaml`
  - `timestamp_column: Date`
  - `duration_column: aHT_Seconds`
  - `frequency: W`
  - `aggregation: median`
  - `seasonal_period: 52`

## 7. Final validation result

```
records: 5,366  (unparseable timestamps dropped: 0)
date range: 2021-01-01 -> 2025-12-31
duplicate rows: 0
duration integrity: 0 NaN, 0 negative, 0 zero

tasks per W-period: mean 20.5, min 8, max 46
active periods below 5 tasks: 0.0%

series length: 262 weeks, missing periods: 0.0%, longest gap: 0

usable observations: 262 -> train 209 / test 53
seasonal period 52: 4.0 cycles in training

CHECKS: {"thin_periods_ok": true, "gaps_ok": true, "train_ok": true, "test_ok": true, "seasonal_ok": true}
VERDICT: PASS
```

## 8. Documented limitation / scope tradeoff

The final series represents **aggregate EMEA support-operations handle time**,
blended across all queues, sites, and channels within the region — not a single
queue's behavior in isolation, unlike the manufacturing dataset's single-machine
log. This is a deliberate breadth-for-density tradeoff and should be stated
explicitly in the methods write-up rather than presented as a homogeneous
single-process series.
