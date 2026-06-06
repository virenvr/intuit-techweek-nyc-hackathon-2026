# SMB Underwriting Hackathon — PRD (Deliverables A–C)

> **Product Requirements Document** for internal team alignment before v2 changes.  
> **Authority:** Official format/rules → [`README.md`](README.md). Strategy/math depth → [`goals.MD`](goals.MD).  
> **Status:** Day 2 — v1 shipped for A & B; v2 planning in progress  
> **Last updated:** 2026-06-06

---

## Table of contents

1. [Purpose & scope](#1-purpose--scope)
2. [Goals](#2-goals)
3. [Current state (v1 baseline)](#3-current-state-v1-baseline)
4. [External constraints](#4-external-constraints)
5. [Shared principles & design decisions](#5-shared-principles--design-decisions)
6. [Interfaces & contracts](#6-interfaces--contracts)
7. [Deliverable A — requirements & plan](#7-deliverable-a--requirements--plan)
8. [Deliverable B — requirements & plan](#8-deliverable-b--requirements--plan)
9. [Deliverable C — requirements & plan](#9-deliverable-c--requirements--plan)
10. [Success metrics](#10-success-metrics)
11. [Work phases, owners & order](#11-work-phases-owners--order)
12. [Risks & open questions](#12-risks--open-questions)
13. [Explicit non-goals](#13-explicit-non-goals)
14. [Decisions log](#14-decisions-log)

---

## 1. Purpose & scope

### Purpose

Align the team on **what we are building**, **what “done” means**, and **in what order** before making major changes to Deliverables A, B, and C.

This PRD exists because v1 runs end-to-end but outputs (especially A bounds and approve patterns) do not yet look like production underwriting. We need a shared contract so “fix the math” becomes concrete, testable work — not ad-hoc rewrites.

### In scope

| Deliverable | File | Scope |
|-------------|------|--------|
| **A** | `submission_A_decisions.csv` | PD, approve/decline, 90% PD intervals |
| **B** | `submission_B_trajectory.csv` | Cohort timing curves (169-row grid) |
| **C** | `submission_C_counterfactuals.csv` | Causal what-if PDs (~900 queries) |
| **Shared** | `src/`, runners, docs | Features, cohort logic, validation pipeline |

### Out of PRD scope (see [§13](#13-explicit-non-goals))

Deliverable D (writeup PDF) is referenced only where A–C choices must match the narrative. D has its own template and review criteria.

### Related documents

| Document | Role |
|----------|------|
| [`README.md`](README.md) | Official submission spec (source of truth for format) |
| [`goals.MD`](goals.MD) | Team bible — NPV math, strategy, ops checklist |
| [`docs/deliverable_a_methodology.md`](docs/deliverable_a_methodology.md) | Auto-generated A run log |
| [`docs/deliverable_b_methodology.md`](docs/deliverable_b_methodology.md) | Auto-generated B run log |

---

## 2. Goals

### North star

Build one **coherent SMB lending system** — not four disconnected models — that maximizes **portfolio profit**, models **default timing**, answers **causal what-ifs**, and produces **honest 90% uncertainty bands**.

### What winning means (by deliverable)

| Dimension | Deliverable | Scoring | Target |
|-----------|-------------|---------|--------|
| Profit | **A** | Auto | Approve set with strong realized P&L on scored population |
| Timing | **B** | Auto | Cumulative default **curves** match when defaults happen |
| Calibration | **A, B, C** | Auto | ~90% interval coverage with **reasonable width** |
| Causality | **C** | Auto | Interventional `do(X=x)` accuracy |
| Reasoning | **D** | Human | Section 3 (causal) strongest; limits honest |

Exact scoring weights are **not published** by organizers. Internal prioritization (see `goals.MD` §12): profit ≈ 30%, trajectory ≈ 25%, calibration ≈ 20%, causal ≈ 10%, writeup ≈ 15% — **unverified**, for tie-breaking only.

### v2 product goal (this PRD)

Move from **“validator passes”** to **“outputs look like real underwriting and score competitively”**:

- Selective approval policy (not ~95% approve rate)
- Tight, calibrated PD intervals (not mostly [0, 1])
- B curves that track A’s portfolio and validate on timing
- C with a defensible causal method (not naive feature overwrite)

---

## 3. Current state (v1 baseline)

### What works today

| Item | Status |
|------|--------|
| A pipeline | `run_deliverable_a.py` — NPV policy, ablation, isotonic PD, per-row interval attempt |
| B pipeline | `run_deliverable_b.py` — cohort assign, timing models, 169 grid, monotonicity |
| Validator | A + B pass format checks when both present in `submission/` |
| Git hygiene | `.gitignore` excludes `dobby/`, `.venv/`, submissions, extracted CSVs |
| Methodology logs | Auto-written under `docs/` on each run |

### Known v1 gaps (why v2 is needed)

#### Deliverable A

| Gap | v1 evidence | Severity |
|-----|-------------|----------|
| Approve-all-ish policy | ~94.5% approve; τ ≈ −$5,868 | **High** |
| Intervals too wide | Mean width ~0.82; ~67% of rows width > 0.9 | **High** |
| PD at extremes | 362 rows PD=1.0, 111 rows PD=0.0 | Medium |
| Prior lender dependence | Ablation kept `prior_underwriter_score` | Medium |
| Selection extrapolation | PD trained approved-only, applied to all | Medium (document) |

#### Deliverable B

| Gap | v1 evidence | Severity |
|-----|-------------|----------|
| Coupled to loose A | ~12,581 / 13,306 approved → cohorts similar | **High** |
| Train lacks cohort weeks | Per-cohort KM from validation only | Medium |
| Simple intervals | Binomial CI, not calibrated on cells | Medium |
| Val backtest | ~2.2% mean abs error vs empirical — OK v1, not final | Low–Medium |

#### Deliverable C

| Gap | Status |
|-----|--------|
| Not started | **Blocker** for full validator PASS |

### Team assessment (Day 2)

> **Framework math (NPV, decision rule) is sound. Estimation, calibration, and policy tuning are incomplete** — teammate feedback on “shaky bounds and patterns” refers to v1 outputs, not wrong problem formulation.

---

## 4. External constraints

Non-negotiable from [`README.md`](README.md):

### Submission package

- **Exactly four files**, **flat folder**, **exact names**
- `validate_submission.py` must print **`PASS`** or submission may not be scored

| File | Rows |
|------|------|
| `submission_A_decisions.csv` | 13,306 (validation + test) |
| `submission_B_trajectory.csv` | 169 (13 × 13 grid) |
| `submission_C_counterfactuals.csv` | ~900 |
| `submission_D_writeup.pdf` | ≤4 pages body |

### Hard format rules

**A:** `pd_lower_90 ≤ predicted_pd ≤ pd_upper_90`; PD required for **all** rows including declines.

**B:** Within each `cohort_week`, `cumulative_default_rate` **non-decreasing** in `loan_age_weeks`; interval ordering enforced.

**C:** `pd_cf_lower_90 ≤ predicted_pd_cf ≤ pd_cf_upper_90`; one row per `query_id`.

### Data facts

- Outcomes only on **prior-approved + matured** rows (selection bias)
- Test has **no outcomes** — A/B/C scored on hidden truth
- C queries use `do(feature = value)` semantics

---

## 5. Shared principles & design decisions

### Principles (all deliverables)

1. **One pipeline story** — A decisions → B cohort curves → C counterfactuals share features and assumptions.
2. **Profit-first A** — Decisions from `E[NPV]`, not flat PD threshold.
3. **Timing matters** — B and A default-day inputs reflect *when*, not just *if*.
4. **Calibration is scored** — 90% intervals must be meaningful, not `[0, 1]` placeholders.
5. **Causality is explicit in C** — Observational ≠ interventional; writeup §3 must match code.

### Locked design decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | Train PD on `approved_matured_mask` only | Labels exist only for prior-approved matured loans |
| D2 | A decision rule: approve iff `E[NPV] > τ` | Aligns with profit scoring |
| D3 | B uses **our** `decision=1` set per `cohort_week` | README contract |
| D4 | Cohort weeks from `application_timestamp` + `cohort_week_definitions.csv` | Official mapping |
| D5 | Re-run B whenever A approve set changes materially | Coupling contract |
| D6 | Generated files in `submission/` — not committed | `.gitignore`; reproduce from runners |
| D7 | Local env: `.venv` + `requirements.txt` — never commit venv | Cross-platform (Windows + Mac) |

### Open design decisions (resolve in v2)

| ID | Question | Options |
|----|----------|---------|
| O1 | τ constraints | τ ≥ 0; max approve rate on val; profit − λ·PD penalty |
| O2 | `prior_underwriter_score` in A | Keep vs drop for cleaner independent policy |
| O3 | Interval method | Mondrian conformal, decile conformal, quantile PD |
| O4 | C causal approach | Double ML, causal forest, structural buckets by feature group |
| O5 | Non-intervenable C queries | Document + best-effort vs exclude from training signal |

---

## 6. Interfaces & contracts

### A → B handoff

```
submission/submission_A_decisions.csv
  columns: applicant_id, decision, predicted_pd, pd_lower_90, pd_upper_90
       ↓
run_deliverable_b.py merges with validation + test
       ↓
submission/submission_B_trajectory.csv
```

**Rule:** Any PR that changes A `decision` column must either (a) re-run B in same PR, or (b) explicitly note “B stale” in PR description.

### Shared feature contract

- Source: `src/feature_engineering.py` — `build_features()`, `feature_columns()`
- A and B should use **compatible** feature sets (document if they diverge)
- Do **not** leak outcome columns (`default_flag`, `days_to_default`, etc.) into scoring features

### Repo layout

| Path | Purpose |
|------|---------|
| `src/` | Shared library code |
| `run_deliverable_a.py` | A entrypoint |
| `run_deliverable_b.py` | B entrypoint |
| `run_deliverable_c.py` | C entrypoint *(to create)* |
| `submission/` | Generated CSVs (gitignored except `.gitkeep`) |
| `docs/` | Auto methodology logs |

### Run order (local)

```powershell
.venv\Scripts\Activate.ps1
python run_deliverable_a.py --output-dir submission
python run_deliverable_b.py --output-dir submission
python run_deliverable_c.py --output-dir submission   # when built
python validate_submission.py submission
```

---

## 7. Deliverable A — requirements & plan

### Functional requirements

| ID | Requirement |
|----|-------------|
| A-R1 | Output 13,306 rows: all validation + test `applicant_id`s |
| A-R2 | `decision` ∈ {0, 1}; `predicted_pd` ∈ [0, 1] for **every** row |
| A-R3 | 90% intervals bracket point estimate on every row |
| A-R4 | Decisions driven by **expected NPV**, not PD cutoff alone |
| A-R5 | PD model trained on approved+matured history only |
| A-R6 | Methodology log updated on each run (`docs/deliverable_a_methodology.md`) |

### v2 improvement plan (priority order)

1. **Policy** — Constrain τ (e.g. τ ≥ 0) and/or cap approve rate on validation; re-tune for profit under constraint.
2. **Intervals** — Replace wide residual bands with conformal / decile-based per-row intervals; target coverage ~85–95% with median width ≪ 0.82.
3. **PD quality** — Reduce mass at 0/1; review isotonic + clipping; consider holdout for calibration vs τ tuning.
4. **Ablation** — Re-evaluate `prior_underwriter_score` after policy fix.
5. **Recovery & default-day** — Validate predicted dollars / days against validation defaults.

### Acceptance criteria (v2 “done”)

| Criterion | Target |
|-----------|--------|
| Validator | PASS on A |
| Approve rate | 30–70% on val+test *(team judgment; not official)* |
| Interval median width | < 0.35 *(internal)* |
| Val interval coverage | 85–95% on approved+matured val |
| Val realized profit | Beats naive baseline (e.g. approve all, copy prior decision) |
| Visual sanity | PD histogram spread; bounds not mostly touching 0 and 1 |

---

## 8. Deliverable B — requirements & plan

### Functional requirements

| ID | Requirement |
|----|-------------|
| B-R1 | 169 rows; grid keys from `dataset/submission_B_template.csv` unchanged |
| B-R2 | For each `(cohort_week, loan_age_weeks)`: cumulative fraction of **our approved** cohort-`w` defaulted by day `7a` |
| B-R3 | Monotone non-decreasing `cumulative_default_rate` within each cohort |
| B-R4 | 90% intervals bracket point estimate |
| B-R5 | Consumes current A `decision` column |
| B-R6 | Methodology log updated (`docs/deliverable_b_methodology.md`) |

### v2 improvement plan (after A v2)

1. **Re-run with tighter A** — Expect more differentiated cohort approve sets.
2. **Timing model** — Consider single survival model vs 13 binary classifiers; smooth curves pre-cummax.
3. **Intervals** — Calibrate `cdr_*` on validation cells; avoid overly narrow binomial bands.
4. **Historical KM** — Document train vs validation cohort coverage; tune `blend_weight` / `MIN_APPROVED_COHORT_SIZE` in `src/constants.py`.

### Acceptance criteria (v2 “done”)

| Criterion | Target |
|-----------|--------|
| Validator | PASS on B |
| Monotonicity | Zero validator errors |
| Val backtest MAE | < 2% mean abs error vs empirical *(maintain or improve)* |
| Age-13 CDR | Within ~3 pp of val empirical per cohort on average |
| Curve shape | Visible rise week 1 → 13; no long flat plateaus unless data-supported |

### Tunable failsafes (v1 — do not remove without replacement)

Located in `src/constants.py` and `TrajectoryModels`:

- `MIN_APPROVED_COHORT_SIZE`
- `blend_weight`
- `MIN_INTERVAL_HALF_WIDTH`, `INTERVAL_Z_SCORE`

---

## 9. Deliverable C — requirements & plan

### Functional requirements

| ID | Requirement |
|----|-------------|
| C-R1 | 900 rows; one per `query_id` in `dataset/intervention_queries.csv` |
| C-R2 | Predict PD under `do(feature = intervention_value)`, rest fixed |
| C-R3 | 90% intervals bracket `predicted_pd_cf` |
| C-R4 | Methodology documented for writeup §3 |
| C-R5 | Intervenable features flagged in `data_dictionary.csv` handled with care; non-intervenable queries documented |

### Proposed approach (v1 — to implement)

1. **Baseline (not sufficient alone):** feature overwrite + re-predict — for comparison only.
2. **Target v1:** Group features by type (self-reported, bank feed, bureau, platform); use **double ML** or **causal forest** on intervenable continuous treatments where feasible.
3. **Intervals:** Same conformal philosophy as A (shared helper if possible).
4. **Sanity checks:** Directional PD shifts on validation for known risk-increasing interventions.

### Acceptance criteria (v1 “done”)

| Criterion | Target |
|-----------|--------|
| Validator | PASS on C |
| Coverage | Intervals valid on all rows |
| Writeup alignment | §3 describes method honestly, including naive baseline limitations |
| Spot-check | ≥80% of manual directional checks on val behave sensibly *(internal)* |

### Dependencies

- Stable **PD model** from A (may share `src/models.py` infrastructure)
- Feature list and preprocessing from `src/feature_engineering.py`

---

## 10. Success metrics

### Auto-scoring proxies (what we can measure locally)

| Metric | Deliverable | How to compute | v1 baseline | v2 target |
|--------|-------------|----------------|-------------|-----------|
| Validator PASS | All | `validate_submission.py` | A+B pass; C missing | All PASS |
| Approve rate | A | `decision.mean()` | 94.5% | 30–70% |
| Interval width | A | `pd_upper_90 - pd_lower_90` | mean 0.82 | mean < 0.40 |
| Interval coverage | A | val approved+matured: y ∈ [lower, upper] | 89.7% | 85–95% |
| Val realized profit | A | `portfolio_realized_profit(validation, decisions)` | $35.2M | Beat baselines *(not absolute)* |
| Trajectory MAE | B | vs empirical CDR on val approved | ~2.2% | ≤ 2.2% |
| C curve monotonicity | B | validator + manual | PASS | PASS |
| C directional sanity | C | manual / val spot-checks | N/A | TBD |

### Baselines to beat (internal benchmarks)

| Baseline | Description |
|----------|-------------|
| Approve all | `decision = 1` everywhere |
| Prior policy | Copy `prior_decision` where available |
| Flat PD threshold | Decline if PD > fixed cutoff (legacy bad approach) |
| Naive C | Feature overwrite only |

---

## 11. Work phases, owners & order

### Phase map

```
Phase 0 ✅  v1 A + B pipelines, gitignore, docs
Phase 1 🔄  PRD agreed + A v2 (policy + intervals)     ← WE ARE HERE
Phase 2 ⏳  B v2 rerun on new A
Phase 3 ⏳  C v1 + intervals
Phase 4 ⏳  D writeup draft (§1–§3 while building C)
Phase 5 ⏳  Full validator PASS + upload
```

### Recommended ownership

| Area | Primary | Support |
|------|---------|---------|
| A policy + PD + intervals | Teammate | Review |
| B survival + grid | You | Sync on A handoff |
| C causal + writeup §3 | TBD / shared | Both |
| Validator + submission QA | Either | Before upload |

### Merge / sync rules

1. No merge to `master` that breaks `run_deliverable_a.py` or validator for A.
2. If `submission_A_decisions.csv` logic changes → re-run B before calling sprint done.
3. Update [Decisions log](#14-decisions-log) when O1–O5 are resolved.

---

## 12. Risks & open questions

### Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Approve-all local optimum on val | Bad test P&L | τ constraints; approve-rate cap |
| Wide intervals score poorly | Calibration penalty | Conformal v2; tune on val |
| Selection bias on declined PD | Wrong NPV at margin | Document; optional IPW later |
| A–B–C inconsistency | Human + auto score loss | This PRD + single feature pipeline |
| Time crunch on C | Missing deliverable | Start C parallel once A policy frozen |
| Overfitting validation | False confidence | Holdout fold for τ vs calibration |

### Open questions

- [ ] **O1:** Final τ constraint strategy?
- [ ] **O2:** Keep `prior_underwriter_score` after A v2?
- [ ] **O3:** Shared conformal module for A/B/C?
- [ ] **O4:** Which causal method for C in remaining time?
- [ ] **O5:** Submission deadline / final upload time?
- [ ] **O6:** Who owns Deliverable D PDF export?

---

## 13. Explicit non-goals

These are **intentionally out of scope** for this PRD and v2 unless time remains after C + validator PASS:

| Non-goal | Reason |
|----------|--------|
| Full reject inference / IPW | High complexity; document as limitation in D |
| Publishing exact scorer replication | Weights unknown |
| Committing datasets, venvs, or submission CSVs | `.gitignore` policy |
| Perfect actuarial NPV (daily cash-flow simulation) | Diminishing returns vs policy + calibration |
| User personas / market analysis PRD sections | Overkill for hackathon |
| Locked hyperparameter search infrastructure | Manual tuning sufficient |
| Step 9 “narrative consistency audit” as gate | Deferred; D covers story |
| Force-push to official Intuit `origin` | Team repo only unless instructed |

---

## 14. Decisions log

| Date | ID | Decision | Notes |
|------|-----|----------|-------|
| 2026-06-05 | D1–D7 | See [§5](#5-shared-principles--design-decisions) | Carried from v1 |
| 2026-06-06 | PRD-1 | Adopt this PRD before major A/B/C changes | Team Day 2 alignment |
| 2026-06-06 | — | v1 A/B accepted as baseline, not final | Teammate review: bounds/patterns weak |

*Add a row when O1–O6 are resolved.*

---

## Appendix: Quick reference commands

```powershell
# Setup (once per machine)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Generate deliverables
python run_deliverable_a.py --output-dir submission
python run_deliverable_b.py --output-dir submission

# Validate
python validate_submission.py submission
```

---

*End of PRD — update when phases complete or decisions change.*
