from pathlib import Path

import joblib
import pandas as pd

from feature_config import (NUMERIC_FEATURES,CATEGORICAL_FEATURES,)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "pl_propensity_model.joblib"

def predict_customer(customer_data):
    model = joblib.load(MODEL_FILE)
    feature_columns = (NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    customer_df = pd.DataFrame([customer_data])[feature_columns]
    probability = model.predict_proba(customer_df)[0][1]
    return probability
if __name__ == "__main__":
    sample_customer = {
        "age": 32,
        "monthly_income": 65000,
        "tenure_months": 24,
        "app_sessions_30d": 15,
        "product_views_30d": 20,
        "loan_page_visits_30d": 5,
        "days_since_last_login": 2,
        "existing_products": 1,
        "delinquent_flag": 0,
        "emi_card_blocked": 0,
        "engagement_score": 3.5,
        "customer_type": "individual",
        "income_band": "Medium",}
    probability = predict_customer(sample_customer)
    print(f"Personal Loan propensity: "f"{probability:.2%}")