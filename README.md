# B2B Marketing Propensity Model - Clean-Room Demo

> **Context for reviewers:** This is a clean-room, non-confidential demonstration project built
> on a synthetic dataset. It mirrors the modeling workflow I apply in production work, which I
> cannot share due to confidentiality. The goal is to show end-to-end Python/ML fluency -
> from data prep through scoring and drift monitoring.

---

## Business Problem

In B2B marketing, campaign budgets are finite. Rather than contacting all accounts, marketing
teams rank accounts by their likelihood to respond and target the top X% - maximizing conversion
rate per dollar spent.

This model predicts `response_flag` (did a B2B account convert within 30 days of campaign
exposure?) and outputs a ranked probability score that can drive prioritization decisions.

---

## Project Structure

```
propensity-modeling-mini/
├── data/
│   └── raw/                    # Synthetic B2B dataset (generated, not real)
├── models/
│   ├── model.joblib            # Serialized sklearn Pipeline
│   ├── metrics.json            # AUC, KS, Brier score
│   └── model_metadata.json     # Training run metadata
├── notebooks/
│   └── 01_build_train_eval.ipynb   # End-to-end walkthrough
├── reports/
│   ├── calibration_curve.png   # Reliability diagram
│   ├── lift_by_decile.png      # Decile lift chart
│   ├── eda_binning_tables.csv  # Response rate by feature bin
│   ├── model_coefficients.csv  # Feature coefficients (ranked by |coef|)
│   └── scored_output.csv       # Full dataset with predicted probabilities
├── src/
│   ├── data.py                 # Data loading + feature engineering
│   ├── eda_binning.py          # EDA: response rate by quantile bin
│   ├── features.py             # sklearn Pipeline + ColumnTransformer
│   ├── train.py                # Model training + artifact serialization
│   ├── evaluate.py             # AUC / KS / calibration / coefficients
│   ├── monitor.py              # PSI-based score drift detection
│   └── score.py                # Batch scoring on new data
├── README.md
└── requirements.txt
```

---

## Modeling Approach

**Algorithm:** Logistic Regression with L2 regularization (intentional baseline choice - see note below)

**Feature engineering** (in `src/data.py`):
- `renewal_window_flag` - account within 3 months of contract end (high-signal business rule)
- `high_engagement_flag` - ≥5 website visits in past 30 days
- `large_account_flag` - company size ≥800 employees
- `revenue_per_employee` - proxy for account health / complexity
- Binned versions of `company_size`, `website_visits_30d`, `contract_remaining_months` (EDA-informed)

**Preprocessing pipeline** (in `src/features.py`):
- Numeric features → `StandardScaler`
- Categorical/binned features → `OneHotEncoder(drop='first')`
- All steps wrapped in `sklearn.Pipeline` for safe train/test separation and deployment

---

## Evaluation Outputs

| Metric | Purpose |
|---|---|
| AUC (train vs. test) | Discrimination; gap flags overfitting |
| KS Statistic | Separation between responders and non-responders |
| Lift by Decile | Directly actionable for campaign targeting decisions |
| Brier Score | Calibration quality (probability accuracy) |
| Calibration Curve | Visual reliability diagram |
| Model Coefficients | Interpretability / feature importance |

Reports are saved to `reports/` on each run.

---

## Drift Monitoring

`src/monitor.py` computes **Population Stability Index (PSI)** between training and scoring
score distributions. PSI > 0.2 triggers a warning, indicating model recalibration may be needed.

In production, this check would run on a schedule (e.g., post each campaign wave) with results
surfaced in a monitoring dashboard. 

---

## Quickstart

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt

python -m src.train      # Train model, save artifacts
python -m src.evaluate   # Generate metrics + reports
python -m src.monitor    # Check score distribution drift
python -m src.score      # Batch score full dataset
```

Or explore end-to-end in the notebook:
```
notebooks/01_build_train_eval.ipynb
```

---

## Why Logistic Regression as the Baseline

Logistic Regression is the deliberate starting point for propensity modeling because:
1. **Coefficients are directly interpretable** - easy to explain to marketing stakeholders
2. **Scores are rank-orderable** - sufficient for top-X% targeting decisions
3. **Fast to train and audit** - establishes a performance floor before adding complexity

Tree-based models (XGBoost, LightGBM) can improve lift, but introduce complexity in monitoring and interpretation - a tradeoff worth evaluating once the baseline is validated.

**For illustration, targeting the top 20% of ranked accounts increases response rate from ~19.6% (baseline) to ~27% (~1.4x lift).** - demonstrating how the model directly informs campaign prioritization under budget constraints.

**Natural next steps:**
- Tree-based model comparison (lift improvement vs. interpretability tradeoff)
- Time-based train/test split to simulate campaign wave validation
- Threshold optimization based on campaign size and contact budget
- Score monitoring scheduled post each campaign wave

---

## Stack

`Python` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `joblib`

In a cloud production environment, this would typically integrate with object storage, scheduled batch scoring jobs, and a monitoring layer tied to a reporting dashboard.
