# Dynamic Homepage Personalization Engine
A beginner-friendly, interview-oriented portfolio project inspired by the themes in your resume.

> **Important:** This is a new demonstration project using synthetic data. It is not the original internal implementation from any employer. All labels and outcomes are simulated and must not be presented as real business results.

## Project roadmap
1. **Data foundation (current step):** create 10,000 synthetic customers, document the schema, validate data, and run exploratory data analysis.
2. Feature engineering and leakage-safe train/validation/test split.
3. Train and compare XGBoost propensity models for Personal Loan (PL), SME, and B2B.
4. Build six-cohort decision logic: PL, SME, B2B, DMS, Engagement, Other.
5. Add rule-based eligibility, fallbacks, tests, and explanation traces.
6. Serve decisions with FastAPI and build an interactive Streamlit demo.
7. Add experiment tracking, monitoring, Docker, and interview preparation.

## Requirements
- Python 3.10 or newer recommended
- Packages listed in `requirements.txt`

## Setup
### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Step 1
From this project folder:
```bash
python src/validate_data.py
python src/eda.py
```

The EDA script writes summary files to `reports/` and prints target prevalence and basic statistics. It does not train models yet.

## Data dictionary
| Column | Meaning |
|---|---|
| `customer_id` | Synthetic unique customer key |
| `age` | Customer age in years |
| `customer_type` | `individual`, `small_business`, or `enterprise` |
| `monthly_income` | Synthetic monthly income in local currency units |
| `tenure_months` | Relationship tenure in months |
| `app_sessions_30d` | App sessions in the last 30 days |
| `product_views_30d` | Product views in the last 30 days |
| `loan_page_visits_30d` | Loan-page visits in the last 30 days |
| `days_since_last_login` | Days since last login |
| `existing_products` | Count of existing products |
| `delinquent_flag` | Synthetic delinquency indicator (0/1) |
| `emi_card_blocked` | Synthetic EMI-card block indicator (0/1) |
| `eligible_pl`, `eligible_sme`, `eligible_b2b` | Synthetic product eligibility flags |
| `applied_pl_90d`, `applied_sme_90d`, `applied_b2b_90d` | Simulated target labels: application within a 90-day outcome window |

## Important modeling note
The `applied_*_90d` columns are targets, **not input features**. Eligibility flags and any features derived from future outcomes need careful treatment. In Step 2 we will define a prediction-time feature set and use a leakage-safe split. For a production-grade causal evaluation, offline metrics alone are not enough; an A/B test or other appropriate experimental design is needed to measure incremental impact.

## Interview explanation for Step 1
"I started by creating a reproducible synthetic customer dataset with profile, engagement, eligibility, and historical outcome fields. I documented the schema and added validation checks for row counts, unique IDs, data types, ranges, missing values, and binary flags. I also explored target prevalence before training, because class imbalance affects metric selection and model evaluation."

Only use that explanation to describe this portfolio implementation after you have run and understood the code.
