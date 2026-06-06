# Deliverable C — Implementation Steps (v1)

> **Purpose:** Step-by-step plan for counterfactual PD (`do(feature = v)`).  
> **Authority:** [`PRD.md`](../PRD.md) §9, [`goals.MD`](../goals.MD) §3 Goal 3, official [`README.md`](../README.md).  
> **v1 focus:** Contract correctness + honest intervention logic + failsafes for traps.

---

## What Deliverable C is

For each of **900 queries** in `dataset/intervention_queries.csv`, predict:

```
P(default | do(feature = v), all other features fixed at applicant's values)
```

Output: `submission_C_counterfactuals.csv` with `query_id`, `predicted_pd_cf`, `pd_cf_lower_90`, `pd_cf_upper_90`.

---

## Traps & pitfalls (read before coding)

| Trap | Why it matters | v1 handling |
|------|----------------|-------------|
| **Naive overwrite** | Setting column + re-predict ≠ causal `do()` | Intervenable: structured `do()` path + side effects; document limits |
| **Intervenable vs not** | 18 features `intervenable=True` in data dict; **174/900 queries** hit non-intervenable fields | Non-intervenable path: blend toward observational PD + wider intervals |
| **Bank-feed nulls** | Bank columns null when `has_linked_bank_feed=False` | Setting bank-feed feature → force `has_linked_bank_feed=True` |
| **Derived columns** | `requested_amount_to_observed_revenue` not rebuilt if raw inputs change | Recompute after intervention |
| **Engineered ratios** | `daily_draw_burden`, debt/revenue ratios depend on intervened values | Always run `build_features()` after raw intervention |
| **Selection bias** | PD trained on approved+matured only | Same as A; document extrapolation in writeup §3 |
| **Confounding** | High revenue correlates with good bureau scores we hold fixed | v1: hold-other-features-fixed; v2: double ML / causal forest |
| **Categorical interventions** | e.g. `owner_personal_credit_band`, `application_channel` | Set value directly; no ordinal assumption |
| **Same applicant, multiple queries** | 300 applicants × ~3 queries | Lookup once per applicant; batch by query |

---

## Step 1 — Load contracts & metadata

- [ ] `dataset/intervention_queries.csv` (900 rows)
- [ ] `dataset/data_dictionary.csv` → `intervenable`, `group` per field
- [ ] `validation.csv` + `test.csv` → applicant features (all query IDs ⊆ val∪test)
- [ ] Build applicant lookup: `applicant_id` → feature row

**Gate:** Every `query_id` and `applicant_id` in queries resolves to a row.

---

## Step 2 — Train / load PD engine (shared with A)

- [ ] Train `UnderwritingModels` via `src/models.py` (approved+matured train, calibrate on val)
- [ ] Use same feature pipeline as `src/feature_engineering.py`
- [ ] v1 default: `include_prior_underwriter_score=False` (cleaner causal story; flag to match A)

**Gate:** `predict_pd()` and `pd_intervals()` work on single-applicant DataFrames.

---

## Step 3 — Classify each query (intervention registry)

For each `(feature_name, intervention_value)`:

| Check | Source |
|-------|--------|
| `intervenable == True` | `data_dictionary.csv` |
| Feature group | `self_reported`, `bank_feed`, `bureau_credit`, `platform_engagement`, `application_context` |
| Known non-intervenable in queries | e.g. `prior_loans_count`, `account_age_days`, `platform_active_months`, `sector`, … |

**Gate:** Log counts: intervenable vs non-intervenable queries (expect ~726 vs ~174).

---

## Step 4 — Apply `do(feature = v)` to applicant state

For each query, on a **copy** of the applicant row:

1. Set `feature_name = intervention_value` (hold all else fixed)
2. **If bank-feed feature:** set `has_linked_bank_feed = True`
3. **Recompute** `requested_amount_to_observed_revenue` when amount or observed revenue touched
4. **Do not** change outcome columns (never present in val/test anyway)

Then `build_features()` runs inside the PD pipeline.

**Gate:** Engineered ratios change when e.g. `stated_annual_revenue` changes.

---

## Step 5 — Predict counterfactual PD

| Path | When | Prediction |
|------|------|------------|
| **Intervenable** | `intervenable=True` | `predict_pd(models, cf_row)` |
| **Non-intervenable** | `intervenable=False` | Blend: `(1−α)·PD_cf + α·PD_obs` (α = failsafe; ill-defined `do()`) |

Use same isotonic-calibrated PD model for both paths.

**Gate:** All `predicted_pd_cf` ∈ [0, 1].

---

## Step 6 — 90% intervals

- [ ] Base intervals from A's per-row residual model (`pd_intervals`)
- [ ] **Non-intervenable:** multiply half-width by `NON_INTERVENABLE_INTERVAL_MULTIPLIER`
- [ ] Enforce `pd_cf_lower_90 ≤ predicted_pd_cf ≤ pd_cf_upper_90`

**Gate:** Validator interval ordering passes on all 900 rows.

---

## Step 7 — Sanity checks (local, pre-submit)

- [ ] Row count = 900; all `query_id`s present exactly once
- [ ] Directional spot-checks on validation (optional v1): higher `aggregate_credit_utilization` → PD not decreasing absurdly
- [ ] Compare intervenable vs non-intervenable mean interval width (non-intervenable should be wider)

---

## Step 8 — Write submission & validate

```powershell
python run_deliverable_c.py --output-dir submission
python validate_submission.py submission
```

- [ ] `submission/submission_C_counterfactuals.csv`
- [ ] `docs/deliverable_c_methodology.md` (auto log for writeup §3)

**Gate:** Full validator PASS (with A, B, C; D PDF still warning until export).

---

## v1 failsafes (tunable in `src/constants.py`)

| Constant | Default | Purpose |
|----------|---------|---------|
| `NON_INTERVENABLE_BLEND` | 0.25 | Pull non-intervenable CF toward observational PD |
| `NON_INTERVENABLE_INTERVAL_MULTIPLIER` | 1.5 | Wider uncertainty when `do()` is ill-defined |
| `MIN_CF_INTERVAL_HALF_WIDTH` | 0.03 | Floor on interval width |

---

## v2 improvements (after contract works)

1. Double ML / causal forest on intervenable continuous features  
2. Shared conformal module with A  
3. Sensitivity analysis on non-intervenable queries  
4. Re-train PD after A v2 policy fix  

---

## File map

| File | Role |
|------|------|
| `src/interventions.py` | Registry, `apply_do()`, derived-column recompute |
| `src/counterfactuals.py` | Batch prediction, interval logic |
| `run_deliverable_c.py` | CLI entrypoint |
| `docs/deliverable_c_methodology.md` | Auto-generated run log |

---

*End of steps — see `run_deliverable_c.py` for implementation.*
