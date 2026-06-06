# SMB Underwriting Challenge — Team Bible

> **Living document.** Update this as our logic improves. When in doubt, the official
> [`README.md`](README.md) and [`dataset/README.md`](dataset/README.md) win on format/rules;
> this file wins on *strategy and execution*.

**Last updated:** 2026-06-05  
**Status:** Active — hackathon in progress

---

## Table of contents

1. [North star](#1-north-star)
2. [What winning means](#2-what-winning-means)
3. [Core goals (A / B / C / D)](#3-core-goals-a--b--c--d)
4. [Mathematical framework & NPV](#4-mathematical-framework--npv)
5. [Data traps we must not ignore](#5-data-traps-we-must-not-ignore)
6. [Deliverable specs (quick reference)](#6-deliverable-specs-quick-reference)
7. [How to win each deliverable](#7-how-to-win-each-deliverable)
8. [Build order & team split](#8-build-order--team-split)
9. [Operational checklist (don't get disqualified)](#9-operational-checklist-dont-get-disqualified)
10. [Common mistakes (where teams lose)](#10-common-mistakes-where-teams-lose)
11. [Writeup (D) priorities](#11-writeup-d-priorities)
12. [Scoring — official vs our assumptions](#12-scoring--official-vs-our-assumptions)
13. [Repo map](#13-repo-map)
14. [Decision log & open questions](#14-decision-log--open-questions)

---

## 1. North star

We are building an **automated SMB lending decision engine**, not a Kaggle classifier.

**Primary objective:** Maximize **realized portfolio profit** on the applications we choose to fund, while producing **calibrated uncertainty**, **correct default timing**, and **defensible causal counterfactuals**.

**One coherent story across A → B → C → D.** Reviewers and scorers should see a single pipeline, not four unrelated hacks.

---

## 2. What winning means

| Dimension | Deliverable | Auto / human | What “good” looks like |
|-----------|-------------|--------------|------------------------|
| Profit | **A** | Auto | Approve loans with positive expected NPV; portfolio P&L beats naive baselines |
| Timing | **B** | Auto | Cumulative default curves match *when* defaults happen, not just *if* |
| Calibration | **A, B, C** | Auto | ~90% intervals that are tight *and* contain truth |
| Causality | **C** | Auto | Interventional `do(X=x)` PDs, not naive feature overwrite |
| Reasoning | **D** | Human | Section 3 (causal) is strongest; assumptions and limits are honest |

---

## 3. Core goals (A / B / C / D)

### Goal 1 — Profit-driven decisioning (Deliverable A)

Do **not** stop at default probability `p̂`. Compute **expected NPV** per loan and approve when `E[NPV | approve] > 0`.

- Use a **profit threshold**, not a flat PD cutoff (e.g. “decline if PD > 20%”).
- `predicted_pd` is required for **everyone**, including declines.
- Our **approve/decline set** feeds directly into Deliverable B.

### Goal 2 — Survival analysis / time-to-event (Deliverable B)

Predict **when** defaults accumulate, not just whether they happen.

- Default at day 5 ≠ default at day 55 (cash-flow timing matters for NPV and trajectory score).
- Use `days_to_default` and survival/hazard methods on approved + matured history.
- B is about **our approved portfolio** per origination cohort week.

### Goal 3 — Causal inference / counterfactuals (Deliverable C)

Estimate `P(default | do(feature = v), rest fixed)`.

- Simple “change the column and re-predict” **will likely fail** on held-out interventional truth.
- Must handle **confounding** and distinguish observational vs interventional prediction.
- Writeup Section 3 must match what we actually built.

### Goal 4 — Methodological defense (Deliverable D)

4-page PDF, substance over polish. **Section 3 weighted most heavily** by reviewers.

---

## 4. Mathematical framework & NPV

### Decision policy

Maximize portfolio value under policy `d` (`d=1` approve, `d=0` decline):

```
max_d  E[ Σ_i  d(x_i) · NPV_i ]
```

**Our rule for A:** `decision_i = 1` iff `E[NPV_i | approve] > 0` (with calibrated `p̂` and timing if we model `t*`).

### Loan constants (fixed product terms)

| Symbol | Meaning | Value |
|--------|---------|-------|
| `r` | APR (annualized) | 0.35 |
| `T` | Term | 60 days |
| `F_i` | Origination fee | 0.03 × R_i |
| `R_i` | Requested amount | `requested_amount` |
| `D_i` | Daily ACH draw | R_i · (1 + r·T/365) / T |

### Default definition (from dataset guide)

A funded loan **defaults** if **any** of:

1. **3 consecutive** missed daily draws (success resets consecutive count), OR
2. **6 cumulative** missed draws (never resets), OR
3. **Outstanding balance > 0 at day 90**

`days_to_default` ∈ [1, 90] is the first day any condition is met.

### NPV sketch (refine as we implement)

**If repaid** (`y_i = 0`):

```
NPV_i ≈ F_i + interest collected over repayment period
```

**If defaulted** at day `t_i*` (`y_i = 1`):

```
NPV_i ≈ F_i + partial payments through t_i* − loss + rec_i
```

- `rec_i` = `final_recovered_amount` (learn recovery rate from train/validation defaults).
- **TODO:** Finalize daily cash-flow NPV formula using `D_i`, fee timing, and recovery model.

### Deliverable B quantity

For cohort week `w`, loan age `a` (weeks):

```
CDR_{w,a} = |{ i ∈ A_w : t_i* ≤ 7a }| / |A_w|
```

Where `A_w` = applicants **we approve** whose `application_timestamp` falls in cohort week `w` (see `cohort_week_definitions.csv`).

**Constraint:** For fixed `w`, `CDR_{w,a}` must be **non-decreasing** in `a`.

### Deliverable C quantity

For query `q` on applicant `i_q`, feature `f_q`, intervention value `v_q`:

```
p_q^cf = P(y_{i_q}=1 | do(f_q = v_q), X_{i_q,−f_q} = x_{i_q,−f_q})
```

---

## 5. Data traps we must not ignore

### Selection bias (critical)

| Split | Rows | Outcomes |
|-------|------|----------|
| `train.csv` | 85,340 | Filled **only** for prior-approved **and** matured loans |
| `validation.csv` | 4,489 | Outcomes filled — **use for tuning & calibration** |
| `test.csv` | 8,817 | **No outcomes** — features only |

- Declined and immature loans have **blank** outcome fields.
- We **cannot** naively train `default_flag ~ X` on all rows and expect valid PD on the full population.
- `prior_underwriter` fields are informative but we are replacing that policy — treat carefully (leakage vs benchmark).

### Missingness is real

- Bank-feed columns are null when `has_linked_bank_feed = False`.
- Some bureau/context fields null when logically N/A (see `data_dictionary.csv`).

### Identifiers

- **`applicant_id`** — one application; use for joins in A and C.
- **`business_id`** — same business may have multiple applications; never cross train/val/test.

### Intervenable features (`intervenable = True` in data dictionary)

Use for causal reasoning:

- Self-reported: `stated_annual_revenue`, `stated_time_in_business`, `requested_amount`
- Bank feed: `observed_monthly_revenue_avg_3mo`, `observed_revenue_trend_3mo`, `observed_revenue_volatility`, `observed_cash_balance_p10`, `observed_overdraft_count_3mo`, `payroll_regularity_score`
- Bureau: `aggregate_credit_utilization`, `recent_inquiries_count_6mo`, `existing_debt_obligations`, `owner_personal_credit_band`
- Platform/context: `invoice_payment_delinquency_rate`, `application_channel`, `multi_lender_inquiry_count_30d`

**Warning:** Some C queries intervene on **non-intervenable** columns (e.g. `prior_loans_count`, `account_age_days`). Document our approach in writeup Section 3.

---

## 6. Deliverable specs (quick reference)

Submit **exactly four files**, **flat folder**, **exact names**.

| File | Rows | Key columns |
|------|------|-------------|
| `submission_A_decisions.csv` | 13,306 (val + test) | `applicant_id`, `decision`, `predicted_pd`, `pd_lower_90`, `pd_upper_90` |
| `submission_B_trajectory.csv` | 169 (13×13 grid) | `cohort_week`, `loan_age_weeks`, `cumulative_default_rate`, `cdr_lower_90`, `cdr_upper_90` |
| `submission_C_counterfactuals.csv` | ~900 | `query_id`, `predicted_pd_cf`, `pd_cf_lower_90`, `pd_cf_upper_90` |
| `submission_D_writeup.pdf` | ≤4 pages body | Five sections per template |

### Hard constraints (validator enforces)

- `pd_lower_90 ≤ predicted_pd ≤ pd_upper_90` (A)
- `pd_cf_lower_90 ≤ predicted_pd_cf ≤ pd_cf_upper_90` (C)
- `cdr_lower_90 ≤ cumulative_default_rate ≤ cdr_upper_90` (B)
- B: cumulative rates **monotone non-decreasing** within each cohort week
- All values in **[0, 1]** where applicable
- Every expected `applicant_id` / `query_id` present — no gaps

---

## 7. How to win each deliverable

### A — Decisions & PD

1. Train PD (and optionally timing) on **approved + matured** train rows.
2. Estimate recovery on defaults (`final_recovered_amount`).
3. Compute `E[NPV | approve]` per applicant; set `decision = 1` if positive.
4. **Tune on `validation.csv`** — we have labels there.
5. Calibrate `predicted_pd` and 90% intervals (Platt / isotonic / conformal / quantile — pick one, validate coverage ≈ 90% on val).
6. Export val + test (13,306 rows).

### B — Default trajectory

1. Map each application → `cohort_week` via `application_timestamp` + `cohort_week_definitions.csv`.
2. Fit survival model on historical approved loans (`days_to_default`, censoring at day 90).
3. For each cohort `w`, predict cumulative default by day `7a` for **our approved** set in that cohort.
4. Enforce monotonicity post-process if needed (cummax per cohort).
5. Calibrate `cdr_lower_90` / `cdr_upper_90` on validation cohorts where possible.
6. Fill `dataset/submission_B_template.csv` — do **not** alter grid keys.

### C — Counterfactuals

1. Load `dataset/intervention_queries.csv` — 900 rows, ~3 queries per applicant.
2. For each query: predict PD under `do(feature = intervention_value)`.
3. **Avoid** naive overwrite on confounded features without adjustment.
4. Candidate methods (choose by time/skill):
   - Double ML / causal forest for continuous treatments
   - Separate structural paths for bank-feed vs bureau vs self-reported
   - Sensitivity analysis for non-intervenable query features
5. Sanity-check directional effects on validation before submit.
6. Calibrate CF intervals like A.

### D — Writeup

See [§11 Writeup priorities](#11-writeup-d-priorities). Draft Section 3 **while** building C.

---

## 8. Build order & team split

```
Phase 0 (first 30 min)
  ├── Register team (Google Form → upload link)
  ├── Unzip dataset
  ├── Run validator on placeholder files → know the gate
  └── EDA: selection bias, default rate, days_to_default distribution

Phase 1 — Foundation
  ├── PD model (approved matured train)
  ├── Recovery model
  └── NPV / approval policy on validation

Phase 2 — Parallel tracks
  ├── Track 1: Survival → B trajectories (+ monotonicity)
  ├── Track 2: Causal engine → C (+ writeup §3)
  └── Track 3: Calibration for A, B, C intervals

Phase 3 — Ship
  ├── Fill all four submission files
  ├── validate_submission.py → PASS
  ├── Export submission_D_writeup.pdf
  └── Upload to private link
```

**Suggested roles (adjust for team size):**

| Role | Owns |
|------|------|
| Policy / A | NPV, threshold, PD calibration |
| Survival / B | Cohorts, KM/hazard, CDR grid |
| Causal / C + D §3 | Counterfactuals, confounding story |
| Integration / QA | Validator, IDs, monotonicity, PDF export |

---

## 9. Operational checklist (don't get disqualified)

### Before we code

- [ ] **Registered** on [team form](https://forms.gle/9VPr8e7KNkZyDBUR6) — upload link comes by email
- [ ] Unzipped `dataset/dataset-compressed.zip` → `train.csv`, `validation.csv`, `test.csv`
- [ ] Read `dataset/data_dictionary.csv` and `dataset/README.md`

### Before every submission attempt

- [ ] Four files with **exact** names (see §6)
- [ ] Files in **one flat folder** — no subfolders
- [ ] A has **13,306** rows — all validation + test `applicant_id`s
- [ ] B has **169** rows — template grid unchanged
- [ ] C has **900** rows — all `query_id`s from `intervention_queries.csv`
- [ ] Interval ordering valid on every row (A, B, C)
- [ ] B curves monotone within each `cohort_week`
- [ ] PDF named `submission_D_writeup.pdf`, ≤4 pages body, ≥11pt, ≥0.75" margins

### Validator (must print `PASS`)

```powershell
pip install -r requirements.txt
python validate_submission.py path\to\your_submission_folder
```

- **ERROR = disqualification risk** — fix all errors
- Warnings (e.g. missing PDF during dry run) — resolve before real upload
- Validator uses `expected_ids/` — full dataset not required to validate format

### Final upload

- [ ] `validate_submission.py` → **PASS**
- [ ] Upload all four files to **team private link** from registration email
- [ ] Keep a local backup copy with timestamp

---

## 10. Common mistakes (where teams lose)

| Mistake | Why it hurts |
|---------|--------------|
| Flat PD threshold instead of NPV | A scores portfolio **profit**, not AUC |
| Train on all rows ignoring selection bias | Biased PD; weak writeup |
| B as flat lines or copy-paste across cohorts | Trajectory score rewards **shape** |
| B ignores our A decisions | B is cumulative defaults on **our** approved set |
| Feature perturbation for C | Fails interventional scoring + writeup §3 |
| Absurdly wide/narrow 90% intervals | Calibration scored on A, B, C |
| Only modeling test, forgetting validation in A | Missing 4,489 val rows → validator fail |
| Skip validator until last minute | Typos / ID gaps = **not scored** |
| Writeup Section 3 generic / disconnected from C | Human reviewers weight §3 most |

---

## 11. Writeup (D) priorities

Template: [`submission_D_writeup_template.md`](submission_D_writeup_template.md) → export **`submission_D_writeup.pdf`**

| § | Section | Our focus |
|---|---------|-----------|
| 1 | Problem framing & assumptions violated | Selection bias, censoring, observational data |
| 2 | Methodology | Single pipeline: PD → NPV → survival → CF |
| **3** | **Causal reasoning & counterfactual methodology** | **Highest weight — spend real time here** |
| 4 | Calibration & uncertainty | How we hit ~90% coverage on validation |
| 5 | Limitations & what we'd do differently | Specific, honest gaps |

**Reviewer preferences:** substance over polish; no marketing; no executive summary.

---

## 12. Scoring — official vs our assumptions

### Official (from README)

Automated: portfolio profitability (A), cohort-timing accuracy (B), calibration (A & B), counterfactual accuracy (C).  
Human: writeup (D).  
**Exact weights and formulas are not published.**

### Team working hypothesis (unverified — update if organizers clarify)

Use for **internal prioritization only**:

| Component | Assumed weight | Notes |
|-----------|----------------|-------|
| P&L / profit | ~30% | Driven by A decisions on scored population |
| Trajectory | ~25% | B shape + level |
| Calibration | ~20% | All 90% intervals |
| Causal / C | ~10% | Still worth serious effort — ties to D §3 |
| Writeup | ~15% | Can differentiate top auto-scored teams |

> If our assumed weights are wrong, the build order in §8 still holds: **never sacrifice validator PASS or calibration for a single dimension.**

---

## 13. Repo map

| Path | Purpose |
|------|---------|
| [`README.md`](README.md) | Official submission rules |
| [`dataset/README.md`](dataset/README.md) | Data guide, default definition, cohorts |
| [`dataset/data_dictionary.csv`](dataset/data_dictionary.csv) | Columns, types, intervenable flag |
| [`dataset/intervention_queries.csv`](dataset/intervention_queries.csv) | Deliverable C inputs |
| [`dataset/cohort_week_definitions.csv`](dataset/cohort_week_definitions.csv) | Week 1–13 date ranges for B |
| [`dataset/submission_B_template.csv`](dataset/submission_B_template.csv) | B grid — fill predictions only |
| [`submission_D_writeup_template.md`](submission_D_writeup_template.md) | Writeup skeleton |
| [`validate_submission.py`](validate_submission.py) | Pre-upload gate |
| [`expected_ids/`](expected_ids/) | ID manifests for validator |

---

## 14. Decision log & open questions

> **How to use:** When we change approach, add a dated line here so we don't relitigate old decisions.

### Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-05 | Adopt NPV-based approval (not flat PD cutoff) | Aligns A with profit scoring |
| 2026-06-05 | Treat this file as team bible | Single source for strategy + ops |

### Open questions (resolve as we explore data)

- [ ] Final closed-form NPV formula with daily draws and recovery — validate against train defaults
- [ ] Best PD model family given selection bias (IPW? two-stage? approve-only + calibration)?
- [ ] Survival model choice for B (KM per cohort vs discrete-time hazard vs parametric)
- [ ] C approach: double ML vs causal forest vs structural — what’s feasible in time box?
- [ ] How to handle C queries on **non-intervenable** features?
- [ ] Recovery rate model: flat fraction vs features?
- [ ] Confirm registration email received and submission deadline / timezone

### Experiments backlog

- [ ] Baseline: prior_underwriter decision copied → measure val profit
- [ ] Baseline: approve all → B trajectory upper bound sanity
- [ ] Calibration plot on validation (90% interval coverage for A)
- [ ] Monotonicity fix strategy for B (raw model vs post-hoc cummax)

---

*End of bible — edit freely, commit strategy changes to Decision log.*
