# Deliverable C v1 log (2026-06-06 17:27 UTC)

Deliverable C — causal traps (writeup Section 3)
------------------------------------------------
Queries ask for P(default | do(feature=v), rest fixed). This is NOT the same as
overwriting a column and re-predicting without adjustment — that answers an
observational counterfactual and fails when confounders exist.

v1 handling:
  - Intervenable features (data_dictionary.intervenable=True):
        apply do(v) with side effects (bank-feed link flag, derived ratios),
        then calibrated PD from the same model as Deliverable A.
  - Non-intervenable features (~174/900 queries — e.g. prior_loans_count,
        account_age_days, sector): causal do() is ill-defined; we blend toward
        observational PD and widen intervals (see constants NON_INTERVENABLE_*).

Remaining gaps for v2: double ML / causal forest; explicit confounding control;
reject-inference consistency with Deliverable A selection bias.

## Query breakdown
- Total queries: 900
- Intervenable: 726
- Non-intervenable (failsafe path): 174
- PD model prior_underwriter_score: OFF

## Queries by feature group
- bank_feed: 269
- bureau_credit: 202
- self_reported: 156
- platform_engagement: 129
- application_context: 95
- business_identity: 49

## Top non-intervenable queried features
- account_age_days: 20
- platform_active_months: 19
- days_since_last_inquiry_elsewhere: 16
- employee_count_bucket: 14
- prior_loans_default_count: 13
- vintage_years: 12
- sector: 12
- intended_use_of_funds: 12
- geography_region: 11
- bookkeeping_recency_days: 11

## Intervals
- Method: Wilson score (z=1.645, k=p-hat*n, local n from calibration support)
- Mean interval width (all): 0.120
- Mean interval width (non-intervenable): 0.175

## v1 failsafes
- NON_INTERVENABLE_BLEND, NON_INTERVENABLE_INTERVAL_MULTIPLIER in src/constants.py
