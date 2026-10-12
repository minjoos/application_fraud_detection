# Application Fraud Detection (LightGBM)

Ranking 1M credit-card and cell-phone applications by fraud risk so a limited review team sees the riskiest ones first.
Academic project · USC Marshall, DSO 562 Fraud Analytics · Fall 2024 · individual submission

**Result:** the final LightGBM model captured **61.2% of all fraud within the top 3% highest-risk applications** on out-of-time data (FDR@3%).

---

## Problem

Identity fraud at account opening often reuses or recombines pieces of real identities (SSN, phone, address, name, DOB).
The goal was a score that ranks applications so that reviewing a small top slice catches as much fraud as possible.

## Data

- 1,000,000 synthetic applications (course-provided; **not redistributed in this repo**)
- Fraud rate ≈ 1.44% (highly imbalanced)
- Last two months (~166K applications) held out as an **out-of-time (OOT)** set

## Approach

**1. Data cleaning: removing false identity links**
Placeholder values such as SSN `999999999` (16,935 records), dummy phone numbers, and default addresses/DOBs would make unrelated applicants look like the same person and poison every linkage feature.
I replaced each placeholder with a record-unique value so it can never create a link.

**2. Feature engineering: 1,500+ candidate variables**
- Entity combinations (e.g., name + DOB, address + phone)
- Days-since-last-seen for each identity element and combination
- Velocity: application counts per entity over 0/1/3/7/14/30-day windows
- Relative velocity: short-window vs long-window ratios
- Unique counts: how many distinct values of one element are linked to another (e.g., distinct SSNs per address)
- Max-count and age indicators

**3. Feature selection**
KS-based filter → top 100 candidates → LightGBM forward-selection wrapper (20 ranked variables) → final model uses the top 10.
The strongest signals were address-level velocity and recency (e.g., max applications per address in 30 days, days since the address was last seen).

**4. Model comparison (OOT FDR@3%)**

| Model | OOT FDR@3% |
|---|---|
| Logistic Regression | 0.589 |
| Decision Tree | 0.593 |
| Random Forest | 0.609 |
| Neural Network (MLP) | 0.564 |
| **LightGBM** (n_estimators 80, max_depth 3, lr 0.07, num_leaves 25) | **0.612** |

LightGBM train / test / OOT: 0.585 / 0.576 / 0.612.

**5. Operating cutoff**
A cost-based curve under course-provided assumptions (cost of a missed fraud vs. cost of a false positive) peaked at a 3% review cutoff.

## Metric note

FDR@3% here means **Fraud Detection Rate**: the share of all fraud that falls inside the top 3% of scores.
It is a recall-at-capacity metric, not precision and not accuracy.

## Limitations

- Synthetic data; no production deployment or real loss savings are claimed.
- The OOT set was also consulted during model selection, so 0.612 is not a fully untouched holdout estimate.
- Any dollar figures from the course cost assumptions are simulations, not business results.

## Repository

```
notebooks/
  01_applications_explore_clean.ipynb       # data quality review, placeholder handling
  02_feature_engineering_variables.ipynb    # candidate variable creation
  03_feature_selection_lightgbm.ipynb       # KS filter + forward wrapper experiments
  04_model_comparison_lightgbm_vs_baselines.ipynb  # model tuning, OOT evaluation, cutoff
report/final_report.pdf
```

Notebooks build on course-provided starter code; data-cleaning decisions, feature design, selection experiments, tuning, and the report are my own work.
The dataset is not included. Notebooks expect it at the path set in the first cell.

**Stack:** Python · pandas · NumPy · scikit-learn · LightGBM · Matplotlib
