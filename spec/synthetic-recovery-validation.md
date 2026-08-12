# Spec: Synthetic Data Generation & Recovery Validation

Three components, specified together because each one's contract
constrains the others:
- a **ground-truth schema** (`ground_truth.json`)
- a **generator** that samples synthetic datasets from it
  (`src/generation/generate_datasets.py`)
- a **recovery validator** that checks the pipeline recovers it
  (`src/recover_structure.py`)

## WHAT

Build a subsystem that:

1. **Pre-registers** the exact daily-level data-generating process (DGP)
   for one or more synthetic operational datasets — base level, optional
   deterministic components (e.g. a weekend-dummy effect, an annual
   seasonal cycle), and a stochastic component (AR(1) or ARMA(1,1)) — in a
   single ground-truth file, written *before* any model is fit against the
   data it describes.
2. **Generates** record-level task/event logs (arrivals, durations,
   dropout, categorical attributes) from that DGP using one fixed seed, so
   the run is fully reproducible, and writes them to the same
   directory/filename convention the rest of the pipeline already expects
   (`configs/*.yaml` → `data/raw/<name>.csv`).
3. **Validates recovery**: rebuilds the daily-median duration series using
   the identical aggregation the screening stage (`data_quality.py`) uses,
   then checks whether standard identification steps (ADF, OLS on
   deterministic terms, AR(1)/ARMA(1,1) fitting) recover the planted
   values within stated tolerance, and reports the result.

## WHY

The forecasting pipeline will run against real task-duration datasets
whose true structure is unknown. A positive control — data with structure
known by construction — is needed to demonstrate the identification
pipeline actually recovers real structure, rather than trusting it on
faith. The validator must also *expect and report* one known effect:
per-record noise plus daily-median aggregation attenuates a recovered AR
coefficient below its planted value; this is a documented outcome, not a
failure condition, for datasets where record-level noise is large relative
to the signal.

## DESIGN DECISIONS (trade-offs, made explicit up front)

These are exactly the places where an unspecified contract would let the
generator and validator drift apart. Each is decided here so both
components are built against the same schema from the start.

| Decision | Options considered | Choice | Why |
|---|---|---|---|
| Ground-truth filename/location | versioned filename (`ground_truth_v3.json`) vs. one stable name; `configs/` vs. repo root | **`ground_truth.json` at repo root, one stable name; version tracked in a `"version"` field inside the file (and in git history)** | A versioned filename requires every consumer (`recover_structure.py`, docs) to be updated in lockstep whenever the generator bumps its version; a stable name with an internal version field decouples "which schema" from "where do I look." Root-level (not `configs/`) because it is pre-registered truth about the datasets, not a per-run configuration knob — it sits alongside `spec/` as project-level provenance. |
| Dataset key naming | free-form key vs. key matching the `name:` field in `configs/<dataset>.yaml` | **Ground-truth dataset key == the config's `name:` field, exactly** (e.g. `generated_emergency_dept_stays`) | The recovery validator has to join ground truth to the config/CSV by name; any naming convention that isn't identical to the yaml `name:` field is a silent join bug waiting to happen. |
| CSV output location | `data/<name>.csv` vs. `data/raw/<name>.csv` | **`data/raw/<name>.csv`**, matching the `file:` field already used in `configs/*.yaml` | Keeps one raw-data convention across generated and non-generated datasets; avoids the validator needing dataset-specific path logic. |
| How the validator knows AR(1) vs ARMA(1,1) | infer from which keys are present (`ar1_phi` vs `arma_phi`) vs. an explicit tag | **Explicit `"stochastic_form": "AR1" \| "ARMA11"` field in `latent_series_dgp`** | Key-presence sniffing breaks silently if a future dataset's schema changes shape (e.g. adds a key for documentation purposes only); an explicit tag fails loudly if missing instead of producing a `None` that crashes deep in report formatting. |
| Annual-seasonality phase | validator hardcodes the phase to match the generator's code vs. ground truth declares it | **Ground truth carries an explicit `annual_phase_reference_day` (e.g. `15` for "day of year 15" / mid-January), and the validator reads it instead of hardcoding `(doy - 15)`** | If phase lives only in generator code, a phase mismatch between generator and validator would look like recovery error rather than a spec bug — moving it into the declared contract makes the assumption inspectable and testable. |
| "Is borderline-ADF expected" flag | infer from `len(planted) > 3` vs. explicit boolean | **Explicit `"seasonal_planted": true/false` field** | A structural proxy (counting keys) is fragile against schema growth (e.g. adding a documentation-only key would flip the heuristic). An explicit flag says exactly what the annotation depends on. |
| Reproducibility | seed per dataset vs. one seed for the whole generation run | **One process-level seed, drawn once, used for every dataset generated in the run** (documented, e.g. tied to a citable value) | Matches "no system entropy" constraint and means the whole ground-truth file is reproducible from one number, not N numbers to track. |

## CONSTRAINTS

- No system entropy: exactly one fixed seed drives all sampling for a
  generation run.
- The validator must construct its series identically to
  `data_quality.py` (same resample frequency, same aggregation function),
  so the screening and validation stages evaluate one consistent view of
  each series.
- Ground truth is written before any model-fitting code runs against the
  data it describes (pre-registration is the point of the exercise —
  fitting first and writing "truth" after would defeat it).
- Recovery validation targets only datasets that have a ground-truth
  entry. Real/found datasets with unknown structure are out of scope for
  this stage by construction, not by oversight.

## RISKS

- **Attenuation is expected but not bounded a priori.** The spec commits
  to *reporting* attenuation, not to a specific numeric tolerance, because
  the true attenuation magnitude depends on the record-level noise model
  chosen per dataset. Each dataset's acceptance tolerance must be set
  per-dataset (see below) rather than globally.
- **Explicit schema fields (this spec's fix) still require discipline**:
  nothing prevents a future dataset from omitting `stochastic_form` or
  `annual_phase_reference_day`. The generator and validator should both
  fail loudly (assertion with a clear message) rather than silently
  defaulting, so a missing field is caught at generation time, not
  discovered as a crash three steps downstream.
- **Single global seed** means changing the generation order of datasets
  (e.g. adding a third dataset before the existing two) changes every
  downstream random draw for datasets generated after it, breaking
  reproducibility of previously-cited numbers. Mitigation: derive a
  per-dataset seed deterministically from the global seed and the dataset
  name (e.g. a hash), so adding a dataset doesn't perturb existing ones —
  flagged here as a trade-off the student should confirm, since it adds
  complexity the simpler "one seed, fixed generation order" option avoids.

## SUCCESS / ACCEPTANCE CRITERIA

- Every dataset's `latent_series_dgp` in ground truth declares
  `stochastic_form` and, when seasonality is planted,
  `annual_phase_reference_day` and `seasonal_planted`.
- The generator writes CSVs to `data/raw/<name>.csv` and ground truth to
  `ground_truth.json`, where `<name>` matches the corresponding
  `configs/<name>.yaml`'s `name:` field exactly.
- The validator runs against `ground_truth.json` with no
  path-patching, for every dataset present in it, and produces
  `reports/recovery_validation.txt` plus one JSON summary line per
  dataset.
- For each dataset, the report states a pass/fail against an explicit,
  per-dataset tolerance for: each planted deterministic effect, the
  planted stochastic coefficient(s), and (where seasonality is planted)
  the ADF outcome on the deterministic residual.
- Re-running generation with the same seed reproduces byte-identical CSVs
  and ground truth (reproducibility is directly testable, not assumed).

## TASKS

| # | Task | Depends on | Independently verifiable by |
|---|------|-----------|------------------------------|
| T1 | Write the ground-truth JSON schema (field names, required vs. optional, allowed `stochastic_form` values) as a short schema doc, separate from any dataset's actual values. | — | Schema doc exists and covers every field decided above. |
| T2 | Design each dataset's DGP (base level, optional deterministic terms, stochastic form + parameters, record-level noise/dropout/volume model) against the T1 schema. | T1 | Each dataset's DGP definition validates against the T1 schema by inspection. |
| T3 | Implement the generator: sample datasets per T2 definitions from one fixed seed, write CSVs to `data/raw/`, write `ground_truth.json`. | T1, T2 | Running the generator produces CSVs + ground truth matching the schema; re-running reproduces identical output. |
| T4 | Implement the recovery validator: rebuild series per `data_quality.py`'s method, run ADF/OLS/ARIMA steps driven by the explicit schema fields (not key-sniffing or hardcoded constants). | T1, T3 | Validator runs against the T3 output with no path or constant patching. |
| T5 | Set and document per-dataset acceptance tolerances (deterministic-effect tolerance, stochastic-coefficient tolerance, expected attenuation direction) referenced by the validator's pass/fail reporting. | T2, T4 | Tolerances are written down per dataset, not inferred at run time. |
| T6 | Run the full pipeline (T3 → T4) end to end and confirm every dataset meets its T5 tolerance; record any that don't as a documented, explained exception rather than silently passing. | T3, T4, T5 | `reports/recovery_validation.txt` shows explicit pass/fail per dataset per tolerance. |

**Ordering:** T1 → T2 → T3 → T4 → T5 → T6 (each step's output is required
input for the next; T5 can be drafted in parallel with T4 but must be
finalized before T6's pass/fail judgment is meaningful).

## OUT OF SCOPE

- Extending recovery validation to real/found datasets (no known ground
  truth exists for them, by definition).
- Choosing which real-world domains to eventually forecast — this spec
  covers only the synthetic validation subsystem.
