# Deliverable B v1 log (2026-06-06 15:59 UTC)

## Cohort assignment (Step 1)
- Scoring applicants: 13,306 (validation + test)
- Approved (from A): 9,930 (74.6%)

## Approved per cohort (Step 2)
- Cohort  1: 822 approved
- Cohort  2: 838 approved
- Cohort  3: 812 approved
- Cohort  4: 822 approved
- Cohort  5: 858 approved
- Cohort  6: 787 approved
- Cohort  7: 786 approved
- Cohort  8: 813 approved
- Cohort  9: 743 approved
- Cohort 10: 737 approved
- Cohort 11: 612 approved
- Cohort 12: 637 approved
- Cohort 13: 663 approved

## Grid summary (Steps 5-7)
- CDR at age 13 weeks (day 91) by cohort:
  - Cohort 1: 0.147 [0.127, 0.168]
  - Cohort 2: 0.148 [0.125, 0.171]
  - Cohort 3: 0.145 [0.118, 0.172]
  - Cohort 4: 0.143 [0.114, 0.172]
  - Cohort 5: 0.148 [0.115, 0.180]
  - Cohort 6: 0.144 [0.108, 0.180]
  - Cohort 7: 0.150 [0.110, 0.190]
  - Cohort 8: 0.148 [0.105, 0.190]
  - Cohort 9: 0.148 [0.100, 0.195]
  - Cohort 10: 0.140 [0.090, 0.189]
  - Cohort 11: 0.139 [0.081, 0.197]
  - Cohort 12: 0.144 [0.083, 0.205]
  - Cohort 13: 0.137 [0.075, 0.198]

## v1 failsafes (tune later)
- min_approved_cohort_size: 5
- blend_weight (thin cohorts): 0.35
- Empty cohort -> historical KM only
- Monotonicity: cummax enforced per cohort
