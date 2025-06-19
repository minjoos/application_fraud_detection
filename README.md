# Credit Card & Application Fraud Detection using LightGBM

This repository presents a real-world inspired fraud detection project designed to identify fraudulent applications and transactions using advanced feature engineering and machine learning. Built using 1M+ synthetic records, this project was completed as part of USC’s MSBA program.

## 🧠 Problem Statement

Detect fraudulent credit card applications using complex patterns in applicant and transaction behavior. The key challenge: maximize fraud detection while minimizing false positives and costs.

---

## 📊 Dataset Overview

- **Size**: ~1,000,000 synthetic credit card application records
- **Fields**: Personal info, SSNs, phone numbers, employer data, financial indicators, labels
- **Challenge**: High imbalance + synthetic placeholders (e.g., SSN = 999999999)

---

## ⚙️ Methodology

### 1. Data Cleaning
- Replaced placeholder values (e.g., 999999999 SSNs, missing phone prefixes)
- Removed duplicates, filtered invalid formats
- Validated data types across all fields

### 2. Feature Engineering
- Engineered **1,500+ features**, including:
  - Entity combinations (name + address + DOB, etc.)
  - Velocity features (e.g., applications per user/time window)
  - Recency and frequency metrics (e.g., phone # reuse patterns)
  - Uniqueness counts and fraud history flags

### 3. Feature Selection
- Filter methods (correlation thresholding, chi-squared)
- Wrapper methods (LightGBM-based importance + SHAP)
- Reduced to **100 final features**

### 4. Model Building & Evaluation
- Models tested: Logistic Regression, Decision Tree, Random Forest, MLP, LightGBM
- Final model: **LightGBM**
  - ✅ Best FDR@3% (Fraud Detection Rate at top 3%)
  - ✅ Stable across train/test/OOT sets
- Evaluation:
  - Out-of-Time (OOT) validation
  - Precision, recall, AUC, FDR@3%, cost savings

---

## 📈 Results

| Model           | FDR@3% (OOT) | Comments                   |
|------------------|--------------|-----------------------------|
| Logistic         | 0.472        | Baseline                   |
| Decision Tree    | 0.503        | Overfit-prone              |
| Random Forest    | 0.538        | Slow, marginally better    |
| MLP              | 0.529        | OK performance             |
| **LightGBM**     | **0.612**    | ✅ Final model              |

- 💰 **Estimated cost savings**: $18B/year (based on top 3% flagged strategy)
- 🔍 Significant patterns found in:
  - SSN reuse
  - Phone # patterns
  - Employer anomalies

---

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Scikit-learn, LightGBM)
- Jupyter Notebook
- Matplotlib / Seaborn
- Git / GitHub

---

## 📁 Key Files

- `notebooks/01_applications_explore_clean.ipynb` – Cleaning & validation
- `notebooks/02_feature_engineering_variables.ipynb` – 1,500+ features
- `notebooks/03_feature_selection_lightgbm.ipynb` – SHAP & wrapper methods
- `notebooks/04_model_comparison_lightgbm_vs_baselines.ipynb` – Final model
- `report/final_report.pdf` – Executive summary & impact analysis

---

## 💡 Key Takeaways

- Importance of preprocessing and deep feature engineering in fraud modeling
- How to balance cost-efficiency with detection coverage
- Techniques for reducing noise in highly imbalanced datasets
- Real-world lessons: model interpretability & risk coverage

---

## 📬 Contact

Feel free to reach out if you'd like to discuss this project or see additional notebooks or deployment code.
