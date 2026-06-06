# Deliverable B v1 log (2026-06-06 17:28 UTC)

## Cohort assignment (Step 1)
- Scoring applicants: 13,306 (validation + test)
- Approved (from A): 9,970 (74.9%)

## Approved per cohort (Step 2)
- Cohort  1: 824 approved
- Cohort  2: 840 approved
- Cohort  3: 810 approved
- Cohort  4: 828 approved
- Cohort  5: 870 approved
- Cohort  6: 801 approved
- Cohort  7: 791 approved
- Cohort  8: 808 approved
- Cohort  9: 747 approved
- Cohort 10: 737 approved
- Cohort 11: 615 approved
- Cohort 12: 635 approved
- Cohort 13: 664 approved

## Grid summary (Steps 5-7)
- CDR at age 13 weeks (day 91) by cohort:
  - Cohort 1: 0.148 [0.129, 0.170]
  - Cohort 2: 0.150 [0.131, 0.171]
  - Cohort 3: 0.144 [0.124, 0.165]
  - Cohort 4: 0.145 [0.126, 0.166]
  - Cohort 5: 0.150 [0.131, 0.171]
  - Cohort 6: 0.147 [0.128, 0.169]
  - Cohort 7: 0.152 [0.132, 0.174]
  - Cohort 8: 0.146 [0.127, 0.168]
  - Cohort 9: 0.148 [0.128, 0.171]
  - Cohort 10: 0.140 [0.121, 0.163]
  - Cohort 11: 0.141 [0.120, 0.166]
  - Cohort 12: 0.145 [0.123, 0.169]
  - Cohort 13: 0.137 [0.116, 0.160]

## v1 failsafes (tune later)
- min_approved_cohort_size: 5
- blend_weight (thin cohorts): 0.35
- Empty cohort -> historical KM only
- Monotonicity: cummax enforced per cohort
- Intervals: Wilson score (z=1.645, n = approved per cohort)
