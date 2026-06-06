# Deliverable A evaluation report (2026-06-06 15:59 UTC)

## Classification metrics
- Log loss:    0.4227
- AUC-ROC:     0.7617
- Brier score: 0.1330

## Calibration metrics
- Expected Calibration Error (ECE): 0.0000

## Interval metrics
- Method: wilson (Wilson score, n=51,722, z=1.645)
- Coverage rate (target 88–92%): 2.4%

## Business metrics
- Approval rate: 74.6%
- Mean predicted PD: 0.271
- Portfolio E[NPV] (approved): $6,185,247
- Portfolio realized profit (validation): $24,888,015
- NPV threshold: $0.00

## Reproducibility
- Random seed: 42
- Feature version: src/feature_engineering.py (application-time)
- Model: HistGradientBoostingClassifier + IsotonicRegression
