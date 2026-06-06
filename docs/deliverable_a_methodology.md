# Deliverable A methodology log (2026-06-06 15:59 UTC)

Selection bias (Deliverable D — Section 1)
------------------------------------------
Training labels (default_flag, days_to_default) exist only for loans the PRIOR
lender approved AND that have matured. Declined and immature applications have
blank outcomes.

Our PD model is fit on that approved+matured subset, then applied to all
validation + test applicants (including those we may decline). PD and NPV for
declined segments are therefore extrapolations, not directly observed.

Mitigations in this pipeline:
  - Train PD only on approved_matured_mask rows
  - Isotonic calibration on validation (same label population)
  - ENPV > 0 decision rule with portfolio metrics on validation
  - Wilson score 90% PD intervals (k = p-hat * n, n = approved+matured train size)
  - Ablation: prior_underwriter_score excluded unless it improves val ENPV
  - Per-applicant default-day and dollar recovery models on default history

Remaining gap: no full reject-inference / IPW — document in writeup limitations.

## Ablation: prior_underwriter_score
- Validation portfolio E[NPV] WITHOUT prior score: $6,022,150
- Validation portfolio E[NPV] WITH prior score:    $6,185,247
- Validation realized profit WITHOUT: $23,887,270
- Validation realized profit WITH:    $24,888,015
- Selected config: WITH prior_underwriter_score

## Policy (PRD FR5)
- Decision rule: approve iff E[NPV] > 0.00
- Interval method: wilson (Wilson score, n=51,722, z=1.645, k=p-hat*n)
- Validation interval coverage (binary y in [lower, upper]): 2.4%
- Random seed: 42
