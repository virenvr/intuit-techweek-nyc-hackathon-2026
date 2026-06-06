# Deliverable A evaluation report (2026-06-06 17:24 UTC)

## Classification metrics
- Log loss:    0.4200
- AUC-ROC:     0.7636
- Brier score: 0.1318

## Calibration metrics
- Expected Calibration Error (ECE): 0.0000

## Interval metrics
- Method: wilson (Wilson score, n=51,722, z=1.645)
- Coverage rate (target 88–92%): 1.7%

## Business metrics
- Approval rate: 75.5%
- Mean predicted PD: 0.271
- Portfolio E[NPV] (approved): $6,176,714
- Portfolio realized profit (validation): $25,503,173
- NPV threshold: $0.00

## Reproducibility
- Random seed: 42
- Feature version: src/feature_engineering.py (application-time)
- Model: HistGradientBoostingClassifier + IsotonicRegression
