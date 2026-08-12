# Ground-Truth JSON Schema (`ground_truth.json`)

Companion to [synthetic-recovery-validation.md](synthetic-recovery-validation.md) —
this is the T1 deliverable: field names and required/optional status, kept
separate from any dataset's actual planted values.

## Top level

| Field | Type | Required | Notes |
|---|---|---|---|
| `version` | string | yes | Schema/content version. No filename versioning — this file is always `ground_truth.json` at repo root. |
| `seed` | int | yes | Single global seed driving every dataset generated in the run. |
| `generated_for` | string | yes | Free-text provenance note. |
| `date_range` | `[start, end]` ISO date strings | yes | Calendar span of every generated series. |
| `datasets` | object | yes | Keyed by dataset name — see below. |

## `datasets.<name>`

`<name>` **must equal** the `name:` field in the corresponding
`configs/<name>.yaml` exactly — the recovery validator joins ground truth
to config/CSV by this key.

| Field | Type | Required | Notes |
|---|---|---|---|
| `domain` | string | yes | |
| `duration_column` | string | yes | Must match the CSV's actual column. |
| `timestamp_column` | string | yes | Must match the CSV's actual column. |
| `aggregation` | string | yes | e.g. `"median per D"`. |
| `latent_series_dgp` | object | yes | See below. |
| `record_level` | object | yes | Free-form: how record-level rows are derived from the latent daily series. |
| `dropout` | object | yes | Mechanism + parameters (`{"mechanism": "none"}` if none). |
| `volume` | object | yes | Arrival/job volume model + parameters. |
| `expected_recovery_note` | string | no | Free-text expectation (e.g. attenuation). |

## `datasets.<name>.latent_series_dgp`

| Field | Type | Required | Notes |
|---|---|---|---|
| `form` | string | yes | Human-readable formula. |
| `base_level_minutes` | float | yes | |
| `weekend_effect_minutes` | float | no | Present only if a weekend dummy is planted. |
| `annual_amplitude_minutes` | float | no | Present only if annual seasonality is planted. |
| `annual_peak` | string | no | Human-readable (e.g. `"mid-January"`). |
| `annual_phase_reference_day` | int | **required if** `annual_amplitude_minutes` present | Day-of-year used as the cosine's phase reference. The validator reads this directly instead of hardcoding a phase. |
| `seasonal_planted` | bool | yes | True if any deterministic seasonal/weekly term is planted. Drives the validator's "borderline ADF expected" annotation — not inferred from key count. |
| `stochastic_form` | `"AR1"` \| `"ARMA11"` | yes | Explicit tag; the validator branches on this instead of sniffing which parameter keys are present. |
| `ar1_phi`, `ar1_sigma` | float | **required if** `stochastic_form == "AR1"` | |
| `arma_phi`, `arma_theta`, `arma_sigma` | float | **required if** `stochastic_form == "ARMA11"` | |

## Validation rule

A `latent_series_dgp` missing `stochastic_form`, or missing
`annual_phase_reference_day`/`seasonal_planted` when seasonality is
planted, is a schema violation. Both the generator and the validator must
fail loudly (raise, with a message naming the dataset and missing field)
rather than silently defaulting.

## Acceptance tolerances (T5)

Per-dataset acceptance tolerances are **not** part of this schema — they
are a validator-stage concern (how close is "recovered"), not a
DGP-stage one (what was planted). They are declared explicitly in
`src/recover_structure.py` (`TOLERANCES`), not inferred from the schema at
run time.
