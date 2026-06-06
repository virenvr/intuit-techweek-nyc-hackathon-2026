# Deliverable B v1 log (2026-06-06 07:56 UTC)

## Cohort assignment (Step 1)
- Scoring applicants: 13,306 (validation + test)
- Approved (from A): 12,581 (94.6%)

## Approved per cohort (Step 2)
- Cohort  1: 1,068 approved
- Cohort  2: 1,043 approved
- Cohort  3: 1,020 approved
- Cohort  4: 1,017 approved
- Cohort  5: 1,099 approved
- Cohort  6: 1,016 approved
- Cohort  7: 993 approved
- Cohort  8: 1,017 approved
- Cohort  9: 939 approved
- Cohort 10: 916 approved
- Cohort 11: 776 approved
- Cohort 12: 826 approved
- Cohort 13: 851 approved

## Grid summary (Steps 5-7)
- CDR at age 13 weeks (day 91) by cohort:
  - Cohort 1: 0.219 [0.198, 0.239]
  - Cohort 2: 0.210 [0.189, 0.231]
  - Cohort 3: 0.207 [0.187, 0.228]
  - Cohort 4: 0.203 [0.183, 0.224]
  - Cohort 5: 0.215 [0.194, 0.235]
  - Cohort 6: 0.211 [0.190, 0.233]
  - Cohort 7: 0.214 [0.193, 0.236]
  - Cohort 8: 0.212 [0.191, 0.233]
  - Cohort 9: 0.213 [0.191, 0.235]
  - Cohort 10: 0.199 [0.177, 0.220]
  - Cohort 11: 0.207 [0.183, 0.231]
  - Cohort 12: 0.216 [0.192, 0.239]
  - Cohort 13: 0.208 [0.185, 0.231]

## v1 failsafes (tune later)
- min_approved_cohort_size: 5
- blend_weight (thin cohorts): 0.35
- Empty cohort -> historical KM only
- Monotonicity: cummax enforced per cohort
