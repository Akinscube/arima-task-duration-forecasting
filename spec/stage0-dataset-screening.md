# Spec: Stage 0 Dataset Screening — Column Inspector & Quality Profiler

Two decoupled CLI tools used in sequence when adding a new candidate
dataset to the dissertation's portfolio:
- a **column inspector** (`src/inspect_columns.py`) — exploratory,
  human-facing, run once while authoring a new `configs/<dataset>.yaml`
- a **quality profiler** (`src/data_quality.py`) — automated, repeatable,
  Stage 0 of the experimental pipeline, gates whether a configured dataset
  is fit to model

## WHAT

**A. Column inspector.** Given a raw CSV:
- *Overview mode* (no column named): report every column's dtype, null %,
  and a few sample values; flag any object-dtype column that looks like a
  timestamp (a high share of a sample parses as a date), so a reader can
  shortlist `timestamp_column` candidates without reading the raw file.
- *Candidate-check mode* (`--ts COL`, optionally with `--end COL` and/or
  `--duration COL`): parse the candidate timestamp column and report null /
  unparseable counts and date range; warn when sampled values are
  day/month-ambiguous (e.g. `03-04-2024`) since naive parsing can silently
  pick the wrong convention; if `--duration` is given, report its
  null/negative/zero counts and summary stats; if both `--end` and
  `--duration` are given, check whether the given duration is consistent
  with the end-minus-start derived duration, and if not, estimate which
  unit (seconds/minutes/hours) the given column is actually in.

**B. Quality profiler.** Given a `configs/<dataset>.yaml` (produced using
the inspector's output): load the dataset, resample the duration column to
the configured frequency/aggregation, and check it against four gates —
task volume per period, calendar-gap tolerance, minimum train/test length,
minimum seasonal cycles in the training window — plus integrity checks
(exact duplicate rows, non-positive/NaN durations, an outlier-vs-p99 flag).
Produce a plain-text report, a series-plus-rolling-mean plot, and an exit
code (0 = every gate passed, 1 = at least one failed) so an external loop
can screen many candidate datasets unattended.

## WHY

Not every dataset that has *a* timestamp and *a* duration column is usable
for ARIMA forecasting — insufficient history, too many gaps, or too little
volume per period will produce an unreliable fit. The dissertation needs
one repeatable, objective bar applied identically to every candidate
dataset, so dataset selection is documented and defensible rather than
ad hoc, and an unsuitable dataset is rejected before time is spent
modelling it. Before that bar can be applied, a human has to correctly
identify *which* raw columns are the timestamp and duration in the first
place — a step prone to a specific, easy-to-miss failure mode (ambiguous
`DD-MM` vs `MM-DD` dates silently parsed the wrong way), which the
inspector exists to catch.

## DESIGN DECISIONS (trade-offs, made explicit up front)

| Decision | Options considered | Choice | Why |
|---|---|---|---|
| Timestamp parsing strictness, inspector vs. profiler | inspector matches the profiler's strict `format="ISO8601"` fail-loud parsing vs. inspector uses permissive parsing | **Inspector uses permissive parsing (`errors="coerce"`, no format pinned) deliberately; profiler uses strict ISO8601 parsing deliberately** | The inspector's job is to *surface* ambiguity to a human (range, unparseable count, day/month warning) — a strict parser would just fail-fast on the first bad row and give the human nothing to look at. The profiler's job is to *refuse* ambiguity once a format has supposedly already been chosen — permissive parsing there would silently miscompute the series. Same behavior in both places would defeat one tool's purpose. |
| Duration/unit-consistency check | leave the given-vs-derived comparison as an ad hoc ratio vs. specify exact bands | **Specify explicitly (T2 below): compute `ratio = median(given) / median(derived_minutes)`; classify as same-unit if `0.9 ≤ ratio ≤ 1.1`, seconds-given if `55 ≤ ratio ≤ 65`, hours-given if `1/65 ≤ ratio ≤ 1/55`, else "unrecognized scale — inspect manually"; report the match rate using the correctly-rescaled given values** | An unresolved ratio-and-threshold definition is exactly the kind of thing that drifts into a subtly wrong implementation (e.g. dividing the wrong way, or classifying using a mismatched tolerance) if left to be inferred from code rather than written down first. |
| Profiler thresholds: global vs. per-dataset override | fixed module-level constants for every dataset vs. allow a config to override thresholds | **Fixed global constants (`MIN_TASKS_PER_PERIOD`, `MAX_MISSING_PCT`, `MIN_TRAIN_OBS`, `MIN_TEST_OBS`, `MIN_SEASONAL_CYCLES`, `TRAIN_FRACTION`), no per-config override** | The dissertation needs "passed Stage 0" to mean the same thing for every dataset in the portfolio; per-dataset overrides would let the bar move to fit whichever dataset is being screened, undermining the comparison. If a specific domain genuinely needs a different bar, that is a decision to make explicitly in this spec (raising or lowering the global constant with a stated reason), not a silent per-config knob. |
| Duplicate-row definition | exact full-row duplicates vs. config-declared key-column duplicates | **Exact full-row duplicates only (no config-declared key)** | Keeps the profiler config-schema-free for this check and avoids requiring every dataset to declare a natural key it may not have. Explicitly documented as a limitation (see RISKS) rather than solved now — a dataset with a unique autoincrement ID will never trip this check even if every other field repeats. |
| Coupling between the two tools | profiler calls the inspector's ambiguity check internally vs. fully decoupled manual workflow | **Fully decoupled**: inspector is run by hand while authoring a config; profiler assumes the config's column choices are already correct and only guards against ambiguity via its own strict parsing | Automatically invoking the inspector from the profiler would blur "exploratory, human-judgment tool" with "automated gate," and the profiler's strict ISO8601 parsing already acts as a compensating control if a human skips the inspector and picks an ambiguous format. |

## CONSTRAINTS

- Both tools operate on a single CSV/config per invocation — no batch
  mode; batch screening is an external loop that shells out per dataset
  and inspects the profiler's exit code.
- The profiler must fail loudly (not silently misparse) on ambiguous or
  mixed timestamp formats — this is non-negotiable given the inspector is
  optional/manual.
- The inspector must never write files or mutate the input CSV — it is a
  read-only exploratory tool.
- The profiler's `reports/<name>_quality.txt` and `reports/<name>_series.png`
  output paths, and its exit-code contract (0 = pass, 1 = fail), are fixed
  so an external screening loop can rely on them.

## RISKS

- **Ambiguity-detection coverage is intentionally narrow.** The inspector
  only flags the `DD-MM` vs `MM-DD` pattern (both parts ≤ 12) on a 500-row
  sample. Two-digit years, unusual separators, and locale-specific formats
  are out of scope — stated here as a non-goal, not a silent gap, so a
  future reader doesn't assume the inspector catches every date-format
  hazard.
- **Fixed global thresholds may reject a legitimately usable
  lower-volume dataset** (e.g. a small single-site manufacturing log).
  Mitigation is a spec change (documented threshold adjustment), not a
  runtime override — flagged so the choice is visible if it comes up.
- **Exact-row duplicate check misses near-duplicates and ID-masked
  duplicates.** Accepted as a known limitation per the design-decisions
  table; revisit only if a real dataset in the portfolio is suspected to
  have this problem.
- **The unit-consistency classification bands (T2) are a judgment call**,
  not derived from a formal error-rate analysis — chosen to be wide enough
  to tolerate normal rounding/measurement noise around 1x/60x/1-60x while
  still being decisive. Worth a sanity check against a real seconds- or
  hours-denominated dataset once one is encountered.

## SUCCESS / ACCEPTANCE CRITERIA

- **Inspector, overview mode:** run against a CSV with at least one
  clearly date-like object column and one clearly numeric column; the
  date-like column is flagged, the numeric column is not.
- **Inspector, ambiguity check:** run against a synthetic CSV where every
  sampled date has day ≤ 12 and month ≤ 12 (genuinely ambiguous); the
  day/month warning fires. Run against a CSV where every date has day > 12
  for at least one sampled row; the warning does not fire.
- **Inspector, unit-consistency check:** run against a synthetic
  `--end`/`--duration` pair where the duration column is deliberately in
  seconds; the reported ratio falls in the seconds band and is reported as
  such (not silently treated as minutes).
- **Profiler:** run against four datasets each engineered to fail exactly
  one gate (thin volume / excessive gaps / short history / too few
  seasonal cycles); each run exits 1 and the report names the specific
  failing check. Run against one dataset engineered to pass every gate;
  it exits 0 and the report says PASS.
- **Profiler outputs:** every run writes `reports/<name>_quality.txt` and
  `reports/<name>_series.png`, regardless of pass/fail.
- **End-to-end:** a shell loop over N configs, checking `echo $?` after
  each profiler run, correctly separates PASS from FAIL datasets without
  needing to read any report by hand.

## TASKS

| # | Task | Depends on | Independently verifiable by |
|---|------|-----------|------------------------------|
| T1 | Write down the column-overview and timestamp-ambiguity detection rules precisely: sample size, parse-rate threshold for "looks like a timestamp," and the exact day/month-ambiguity pattern/threshold. | — | Rule doc covers every heuristic used in inspector overview + ambiguity-check output. |
| T2 | Write down the duration/unit-consistency check precisely: the ratio formula and the same-unit/seconds/hours/unrecognized classification bands (see DESIGN DECISIONS table). | — | Rule doc gives an unambiguous formula a developer can implement without guessing. |
| T3 | Implement the column inspector (overview mode; `--ts`, `--ts --end`, `--ts --duration`, `--ts --end --duration` modes) per T1/T2. | T1, T2 | Runs against the SUCCESS CRITERIA fixtures and matches expected flags/warnings. |
| T4 | Write down the profiler's four gate formulas (volume-per-period thinness, gap tolerance vs. seasonal period, train/test/seasonal-cycle minimums) and the two constants decisions from the table (global thresholds, exact-row duplicates) as one threshold reference. | — | Threshold reference lists every constant, its value, and which check uses it. |
| T5 | Implement the profiler (four gates + integrity checks + report + plot + exit code) per T4. | T4 | Runs against the SUCCESS CRITERIA pass/fail fixtures and matches expected exit codes and named failures. |
| T6 | Build the fail-one-gate-at-a-time fixture datasets (four) and the all-pass fixture dataset described in SUCCESS CRITERIA; run both tools against all fixtures and record actual vs. expected outcomes. | T3, T5 | A short results table: fixture → expected outcome → actual outcome, all matching. |

**Ordering:** T1, T2, T4 can proceed in parallel (independent specs for
independent checks). T3 depends on T1+T2; T5 depends on T4. T6 depends on
both T3 and T5 being built.

## OUT OF SCOPE

- Batch/parallel screening across many datasets in one process — handled
  by an external loop, not by either tool.
- Auto-generating `configs/<dataset>.yaml` from the inspector's output —
  the inspector informs a human-authored config; it does not write one.
- Locale-aware date-format detection beyond the single DD-MM/MM-DD
  ambiguity check.
