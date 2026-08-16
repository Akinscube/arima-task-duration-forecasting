# Spec: Analyse and Interpret Results

## GOAL (restated)

Synthesize everything already computed by the prior four stages (EDA,
baselines, ARIMA fitting, model evaluation) into the actual RQ1 and RQ2
answers — a reproducible master evidence table plus a narrative document
that directly addresses both locked research questions, including RQ2's
own aviation-MRO workforce-planning framing — using only evidence that
already exists, not new computation.

## WHAT

1. **Master evidence table** (`src/analyse_results.py`): reads the
   trailing JSON summary lines already written by `eda.py`,
   `baselines.py`, `arima_model.py`, and `evaluate_model.py`'s respective
   cross-domain summary files, merges them on `dataset`, and writes one
   row per dataset covering: EDA diagnostics, ARIMA order-selection
   results, all five methods' MAE/RMSE/MAPE, and the DM win/loss/tie
   counts. No new statistic is computed — this is a join, not an
   analysis.
2. **Narrative document** (`reports/analysis/rq1_rq2_analysis.md`),
   structured as:
   - **RQ1 verdict** — does ARIMA beat each baseline, per dataset and
     overall, quoting `evaluate_model.py`'s DM results verbatim
     (including the "provisional pending supervisor confirmation" status
     — this document does not resolve or soften that caveat).
   - **RQ2: cross-domain performance variation** — connects EDA's
     temporal-property diagnostics (stationarity, seasonality strength,
     dominant lags) to the pattern already visible in three independent
     prior stages (order-selection confidence, MAPE ranking, DM win/tie
     counts) — identifies which domain(s) ARIMA underperforms in and why,
     using only evidence already on record.
   - **Aviation MRO workforce-planning implications** — a discursive
     section connecting the RQ2 evidence pattern to what it implies for
     deploying ARIMA in a workforce-planning tool for a domain like
     aviation MRO, grounded strictly in the four tested domains'
     evidence (no MRO dataset was tested — `CLAUDE.md`: "Aviation MRO is
     motivating background only").
   - **Limitations** — explicitly restates every unresolved caveat
     already on record (DM provisional status, MAE-only significance
     testing, per-dataset-not-global correction, `call_center_emea_2021_2025`'s
     low test-set power, plain-ARIMA-only scope, the pending
     generated-vs-real dataset question) rather than letting them get
     lost by the time conclusions are drawn.

## WHY

- `CLAUDE.md`: "Every implementation choice should trace back to RQ1 or
  RQ2" — this stage is where that finally gets cashed out into an actual
  answer, not just design decisions that gesture at the questions.
- Every prior stage's spec (`baseline-forecasting-models.md`,
  `arima-fitting.md`, `model-evaluation.md`) explicitly deferred
  cross-domain interpretation to "the later 'analyse and interpret
  results' stage" — this is that stage; nothing upstream is finished
  answering RQ1/RQ2 until this exists.
- A reproducible master table (not just prose) means the numbers backing
  every claim in the narrative document can be independently regenerated
  and spot-checked, consistent with this pipeline's examiner-rerunnable
  design throughout.

## DESIGN DECISIONS

| Decision | Options considered | Choice | Why |
|---|---|---|---|
| Deliverable format | Script + narrative doc vs. hand-assembled doc only | **Script + narrative doc** | A deterministic join script keeps the master table independently re-verifiable by rerunning it, rather than a one-off hand assembly that could silently drift from the underlying reports as they're regenerated. The interpretation itself still has to be written prose — no script can produce the "why" or the workforce-planning implications. |
| Aviation MRO discussion | Address substantively vs. leave to the dissertation's own Discussion chapter | **Address substantively, evidence-grounded** | RQ2's locked wording explicitly asks about impact on "workforce planning domains like aviation MRO" — leaving it out would leave RQ2 half-answered by this repo's own artifacts. Grounded strictly in the four tested domains' evidence, not a new MRO-specific claim, consistent with `CLAUDE.md`'s "motivating background only" framing. |
| Master table source | Re-parse each stage's free-text per-dataset reports vs. use only the already-JSON cross-domain summary lines | **JSON cross-domain summary lines only** (`eda_cross_domain_summary.txt`, `baselines_cross_domain_summary.txt`, `arima_cross_domain_summary.txt`, `evaluation_cross_domain_summary.txt`) | Every one of those four files already ends with one `json.dumps(row)` line per dataset — parsing free-text `.txt` reports for the same information would be fragile and duplicate data that's already machine-readable. The master table therefore carries win/loss/tie *counts* per dataset (not full per-baseline p-values); the narrative document cites individual per-dataset evaluation reports directly when a specific p-value is discussed. |
| Modifying prior stages | Extend `evaluate_model.py` to emit more detailed per-pair JSON vs. leave all four prior scripts untouched | **Leave untouched** | This stage's constraint is to be purely additive; changing an already-verified, already-reproducible prior stage's output format to suit this stage's convenience would risk invalidating work already checked and (potentially) already committed. |

## CONSTRAINTS

- Reads only the four prior stages' already-persisted cross-domain
  summary files' trailing JSON lines — no new statistical computation, no
  re-derivation of any metric.
- Does not modify `data_quality.py`, `eda.py`, `baselines.py`,
  `arima_model.py`, `evaluate_model.py`, `recover_structure.py`, any
  spec, or `ground_truth.json`.
- Every claim in the narrative document must trace to a specific existing
  report/number (spot-checkable) — no unsupported assertions.
- The DM "provisional pending supervisor confirmation" status must be
  stated plainly wherever RQ1's verdict is discussed — this stage does
  not upgrade DM's status on its own authority.
- The aviation-MRO section stays strictly evidence-grounded to the four
  tested domains — no fabricated or assumed MRO-specific data.
- Output under `reports/analysis/`, never into any prior stage's report
  directory.

## RISKS

- **Master table could silently go stale** if a prior stage's
  cross-domain summary column names ever change. The loader must fail
  loudly (raise, not silently produce blanks/NaNs) on a missing expected
  key, so drift is caught immediately rather than propagating into the
  narrative document unnoticed.
- **RQ1's verdict is provisional, not final.** If the supervisor declines
  DM or requires a different methodology, this document's RQ1 section
  may need rewriting. Stated plainly at the top of the narrative document,
  not buried in the limitations section alone.
- **Ties are not evidence of equivalence.** `no_significant_difference`
  (especially on `call_center_emea_2021_2025`'s 53-point test set) means
  "not detected as different," not "proven equal" — the narrative
  document must word this carefully throughout, not just once in the
  limitations section, since it's exactly the kind of claim easy to
  overstate when writing a conclusions narrative.
- **The generated-vs-real dataset question is still pending supervisor
  input** (per the message drafted for the model-evaluation stage). The
  narrative document's RQ2/aviation-MRO discussion draws on both real and
  generated datasets' evidence; this is stated as a live open question,
  not treated as already settled.
- **Aviation MRO framing risk**: RQ2 explicitly asks about impact on MRO
  specifically, but none of the four tested datasets is an MRO dataset.
  The discussion must repeatedly signal "by analogy" / "if this pattern
  holds" rather than asserting the finding transfers directly — overreach
  here is the single easiest way for this section to weaken rather than
  strengthen the dissertation's methodological credibility.

## SUCCESS / ACCEPTANCE CRITERIA

- `reports/analysis/master_evidence_table.csv` (and a human-readable
  `.txt`) has exactly one row per of the four final datasets, every
  column populated from a named source file, no blanks.
- Re-running `src/analyse_results.py` reproduces identical output
  (deterministic — it only reads existing files).
- `reports/analysis/rq1_rq2_analysis.md` directly answers RQ1 (with the
  DM-provisional caveat stated up front) and RQ2 (domain-variation
  pattern, the underperforming domain(s) named explicitly, and the
  aviation-MRO workforce-planning implication grounded in that evidence).
- Every numeric claim in the narrative document is traceable to a named
  source report (T5's spot-check must pass).
- Limitations section explicitly lists all five items named in WHAT.

## TASKS

| # | Task | Depends on | Independently verifiable by |
|---|------|-----------|------------------------------|
| T1 | Write `results-analysis-parameters.md`: master table schema/column list, exact source file paths and JSON-line parsing rule, output paths. | — | Doc exists, no blanks, every column traces to a named source field. |
| T2 | Implement the loader: parse the trailing JSON line(s) from each of the four cross-domain summary files, merge on `dataset`, fail loudly on any missing expected key. | T1 | Running against the four datasets' existing summary files produces a merged row per dataset with no missing fields; a deliberately truncated copy of one summary file (scratch test) raises rather than silently producing a blank. |
| T3 | Write the master table (CSV + human-readable `.txt`) and run it end-to-end; verify reproducibility. | T2 | Two consecutive runs produce byte-identical output. |
| T4 | Write the narrative document (RQ1 verdict, RQ2 domain-variation analysis, aviation-MRO discussion, limitations), citing the master table and underlying per-dataset reports directly. | T3 | Document exists with all four named sections, no section empty. |
| T5 | Spot-check every numeric claim in the narrative document against its cited source report. | T4 | A manual pass confirms each cited number matches its source exactly; any mismatch is corrected before this task is considered done. |

**Ordering:** T1 → T2 → T3 → T4 → T5.

## OUT OF SCOPE

- Any new statistical computation, metric, or significance test beyond
  what the four prior stages already produced.
- Resolving DM's provisional status — still pending supervisor
  confirmation, unaffected by this stage.
- Resolving the generated-vs-real dataset question — still pending
  supervisor input.
- A full literature review or discussion beyond what the four tested
  datasets' evidence supports.
- Modifying any prior stage's script, spec, or report format.
