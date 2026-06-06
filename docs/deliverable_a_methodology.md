# Deliverable A methodology log (2026-06-06 06:48 UTC)

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
  - NPV threshold tuned on validation realized profit (not AUC alone)
  - Ablation: prior_underwriter_score excluded unless it improves val profit
  - Per-applicant default-day and dollar recovery models on default history

Remaining gap: no full reject-inference / IPW — document in writeup limitations.

## Ablation: prior_underwriter_score
- Validation profit WITHOUT prior score: $34,884,339
- Validation profit WITH prior score:    $35,162,451
- Selected config: WITH prior_underwriter_score

## Policy tuning
- NPV threshold (tau): $-5,868.52  (approve if E[NPV] > tau)
- Validation interval coverage (binary y in [lower, upper]): 89.7%
