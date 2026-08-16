# RQ1 / RQ2 Analysis

Synthesizes `reports/eda/`, `reports/baselines/`, `reports/arima/`, and
`reports/evaluation/` (via `reports/analysis/master_evidence_table.csv`)
into direct answers to RQ1 and RQ2. No new statistic is computed here —
every number below is cited from an existing report.

## Status banner

**The significance testing behind RQ1's verdict (Diebold–Mariano) is
provisional, pending supervisor confirmation** (`CLAUDE.md`: "NOT
confirmed by supervisor — do not implement as final methodology without
confirmation"; confirmation has been requested, not yet received). Every
win/loss/tie claim below inherits that caveat. If DM is not confirmed, or
a different methodology is required, this section's verdicts may need
re-deriving.

## RQ1 verdict

> *"Is the prediction of the task duration made using an ARIMA model more
> accurate than that done using conventional baseline methods of forecast
> (naïve, historical mean, moving average, exponential smoothing), as
> measured using MAE, RMSE, and MAPE when considering test data taken
> from different operational domains?"*

| Dataset | vs naïve | vs historical mean | vs moving average | vs SES |
|---|---|---|---|---|
| `synthetic_it_support_tickets` | **ARIMA wins** (p=0.000) | tie (p=1.000) | **ARIMA wins** (p=0.001) | tie (p=1.000) |
| `call_center_emea_2021_2025` | **ARIMA wins** (p=0.002) | tie (p=0.753) | tie (p=0.753) | tie (p=0.467) |
| `generated_emergency_dept_stays` | **ARIMA wins** (p=0.000) | **ARIMA wins** (p=0.000) | **ARIMA wins** (p=0.001) | **ARIMA wins** (p=0.007) |
| `generated_job_shop_manufacturing` | **ARIMA wins** (p=0.000) | **ARIMA wins** (p=0.000) | **ARIMA wins** (p=0.000) | **ARIMA wins** (p=0.000) |

(p = Holm-adjusted DM p-value; source: `reports/evaluation/<name>_evaluation.txt`)

**Answer: conditionally yes.** ARIMA never loses to a baseline on MAE, on
any dataset, at any point in this study — no cell in the table above is a
baseline win. It beats the naïve baseline in all four domains, and beats
*every* baseline in two of the four domains
(`generated_job_shop_manufacturing`, `generated_emergency_dept_stays`).
But in the other two (`call_center_emea_2021_2025`,
`synthetic_it_support_tickets`) it only significantly beats naïve — against
the smarter baselines (historical mean, moving average, SES) the
difference is not statistically distinguishable from zero. So RQ1's
answer is not a uniform yes: **ARIMA's accuracy advantage over
conventional baselines is real but domain-dependent**, which is exactly
what RQ2 asks about next.

## RQ2: cross-domain performance variation

> *"How varied is the performance of the ARIMA forecasting technique
> between domains that have differing temporal properties? In what
> domain task duration times series is this technique underperforming,
> and how will this impact the use of ARIMA in workforce planning domains
> like aviation MRO?"*

### The pattern

| Dataset | Seasonal strength (F_s) | ARIMA order | AIC/BIC agree | MAPE | RQ1 win count (of 4) |
|---|---|---|---|---|---|
| `generated_job_shop_manufacturing` | 0.096 (none) | (1,0,1) — matches planted ARMA(1,1) exactly | yes | 2.96% | 4/4 |
| `generated_emergency_dept_stays` | 0.522 (strong) | (5,1,1) | yes | 7.66% | 4/4 |
| `call_center_emea_2021_2025` | 0.331 (borderline) | **(0,0,0)** — no structure found | yes | 10.72% | 1/4 |
| `synthetic_it_support_tickets` | 0.221 (none) | (5,0,2) | **no** — BIC prefers (0,0,0) | 16.68% | 2/4 |

(sources: `reports/eda/eda_cross_domain_summary.txt`,
`reports/arima/arima_cross_domain_summary.txt`,
`reports/arima/arima_order_selection_documentation.md`)

**Seasonality strength alone does not predict where ARIMA wins.**
`generated_emergency_dept_stays` has the *strongest* seasonality of the
four (F_s=0.522) and yet ARIMA sweeps all four baselines there — a plain,
non-seasonal ARIMA (locked design decision, see Limitations) still found
enough exploitable serial structure via a high-order AR term (order 5) to
win decisively. Conversely, `call_center_emea_2021_2025` has middling
seasonality (F_s=0.331, borderline by this study's own threshold) but is
the domain where ARIMA underperforms most clearly.

**What does predict it: whether AIC/BIC order selection finds real
structure at all.** The two domains where ARIMA sweeps all four baselines
are exactly the two where order selection was confident and unambiguous
(AIC and BIC agree, and for job-shop the selected order matches the
*planted* ground-truth process exactly). The two domains where ARIMA only
ties are exactly the two where order selection found weak or contested
structure: `call_center_emea_2021_2025`'s grid search concluded the
best model is `ARIMA(0,0,0)` — literally "predict the training mean,"
no AR or MA terms improve on it — and `synthetic_it_support_tickets` is
the one dataset in the whole study where AIC and BIC actively disagree
about whether the extra structure in (5,0,2) is worth its complexity.

### Where ARIMA underperforms

**`call_center_emea_2021_2025` is the clearest underperforming domain.**
It is the only dataset where ARIMA beats just one of the four baselines
(naïve) and ties the other three, its own order-selection process found
no exploitable AR/MA structure at all, and it has the weakest evidence
overall (only 53 test points — see Limitations). `synthetic_it_support_tickets`
is a secondary, less severe case: it still beats two of four baselines,
but has the worst MAPE of all four datasets (16.68%) and the one
AIC/BIC disagreement in the study, both pointing the same direction —
comparatively little real temporal structure for ARIMA to exploit.

## Aviation MRO workforce-planning implications

RQ2 asks specifically about impact on workforce-planning domains like
aviation MRO. **No MRO dataset was tested in this study** — `CLAUDE.md`
is explicit that aviation MRO is motivating background only — so
everything in this section is inference by analogy from the four tested
domains, not direct MRO evidence, and should be read that way throughout.

The evidence pattern above suggests a workforce-planning system should
**not assume ARIMA is uniformly the right forecasting choice across an
organization's task streams**. `generated_job_shop_manufacturing` and
`generated_emergency_dept_stays` show that where a task-duration series
has strong, exploitable serial persistence — whether from a simple
autoregressive process (job-shop) or from a stronger, weekly-patterned
process that a high-order AR term can still approximate without an
explicit seasonal component (ED) — ARIMA delivers a real, statistically
supported accuracy advantage over the cheap baselines a planner might
otherwise default to. But `call_center_emea_2021_2025` — a real (not
generated) operational dataset, and arguably the closest analogue among
the four to a busy, human-driven service queue such as an MRO
line/hangar — shows the opposite: when a task stream's durations behave
close to noise at the aggregation level used, ARIMA's extra modeling
complexity buys nothing measurable over a simple historical-mean or SES
forecast.

The practical implication for an MRO workforce-planning deployment: **run
this study's own diagnostic sequence (stationarity/ACF/PACF screening,
order-selection confidence) per task type or queue before committing to
ARIMA for it**, rather than deploying one ARIMA-based tool uniformly
across an organization whose different task streams (routine line
maintenance vs. unscheduled heavy-check findings, say) are likely to
differ in temporal properties just as this study's four domains did. A
task stream that screens like `call_center_emea_2021_2025` is evidence
that a cheaper baseline should be preferred there specifically, not proof
that ARIMA is unsuitable for MRO scheduling as a whole.

## Limitations

- **Diebold–Mariano is provisional, pending supervisor confirmation** —
  restated from the Status banner because it is the single most
  consequential open item behind every claim in this document.
- **Only MAE-based significance was tested.** RMSE/MAPE are reported
  descriptively per pair but were not independently significance-tested
  (`spec/model-evaluation.md`'s scope decision) — a dataset could in
  principle show a different pattern under a squared-error-based test.
- **Multiple-comparisons correction is per-dataset, not global.** The
  16 DM tests across all four datasets are not jointly false-discovery
  controlled; the cross-domain pattern above is an informal aggregate of
  four independently-corrected families.
- **`call_center_emea_2021_2025` has only 53 test points** — its three
  "no significant difference" verdicts may partly reflect insufficient
  statistical power, not genuine equivalence between ARIMA and those
  baselines.
- **Plain ARIMA only, no seasonal (SARIMA) term, anywhere** — a locked
  design decision (`spec/arima-fitting.md`), applied uniformly so the
  cross-domain comparison is of domain fit rather than model class. This
  means `generated_emergency_dept_stays`'s win happened *without* ARIMA
  directly modeling its known weekly seasonality — a seasonal model might
  do even better there, which this study does not test.
- **The generated-vs-real dataset mix is a pending question to the
  supervisor** (two of the four datasets are researcher-generated, not
  real). This document draws on both without distinction in the
  cross-domain pattern above; that mix itself has not yet been confirmed
  as methodologically sound by the supervisor.
