from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customers_synthetic.csv"

EXPECTED_COLUMNS = {
    "customer_id", "age", "customer_type", "monthly_income", "tenure_months",
    "app_sessions_30d", "product_views_30d", "loan_page_visits_30d",
    "days_since_last_login", "existing_products", "delinquent_flag",
    "emi_card_blocked", "eligible_pl", "eligible_sme", "eligible_b2b",
    "applied_pl_90d", "applied_sme_90d", "applied_b2b_90d",
}
BINARY_COLUMNS = [
    "delinquent_flag", "emi_card_blocked", "eligible_pl", "eligible_sme",
    "eligible_b2b", "applied_pl_90d", "applied_sme_90d", "applied_b2b_90d",
]

def validate_data(path: Path = DATA_PATH) -> list[str]:
    df = pd.read_csv(path)
    errors = []

    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {sorted(missing_columns)}")

    if len(df) != 10_000:
        errors.append(f"Expected 10,000 rows, found {len(df)}")

    if "customer_id" in df.columns:
        if df["customer_id"].isna().any():
            errors.append("customer_id contains null values")
        if df["customer_id"].duplicated().any():
            errors.append("customer_id contains duplicates")

    if "age" in df.columns and not df["age"].between(18, 100).all():
        errors.append("age contains values outside expected range 18–100")

    if "monthly_income" in df.columns and (df["monthly_income"] < 0).any():
        errors.append("monthly_income contains negative values")

    if "customer_type" in df.columns:
        allowed = {"individual", "small_business", "enterprise"}
        unexpected = set(df["customer_type"].dropna().unique()) - allowed
        if unexpected:
            errors.append(f"Unexpected customer_type values: {sorted(unexpected)}")

    for col in BINARY_COLUMNS:
        if col in df.columns and not set(df[col].dropna().unique()).issubset({0, 1}):
            errors.append(f"{col} must contain only 0/1 values")

    if df.isna().any().any():
        errors.append("Dataset contains missing values")

    return errors

if __name__ == "__main__":
    problems = validate_data()
    if problems:
        print("DATA VALIDATION FAILED")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("DATA VALIDATION PASSED")
    print(f"Rows: 10,000 | Columns: {len(EXPECTED_COLUMNS)}")
    print("Customer IDs are unique; expected ranges and binary flags are valid.")
