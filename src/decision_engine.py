from pathlib import Path

import joblib
import pandas as pd

from feature_config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGETS,
)


# ----------------------------------------
# 1. Project configuration
# ----------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

PRODUCTS = {
    "pl": {
        "name": "Personal Loan",
        "eligibility_column": "eligible_pl",
        "model_file": "pl_propensity_model.joblib",
    },
    "sme": {
        "name": "SME Loan",
        "eligibility_column": "eligible_sme",
        "model_file": "sme_propensity_model.joblib",
    },
    "b2b": {
        "name": "B2B Finance",
        "eligibility_column": "eligible_b2b",
        "model_file": "b2b_propensity_model.joblib",
    },
}

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Illustrative prototype rules. Tune these with validated business policy.
LOW_SESSION_LIMIT = 2
INACTIVE_DAYS_LIMIT = 14


# ----------------------------------------
# 2. Load trained models and thresholds
# ----------------------------------------

def load_models():

    models = {}

    for product_code, config in PRODUCTS.items():

        model_path = MODEL_DIR / config["model_file"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Run train_all_models.py first."
            )

        artifact = joblib.load(model_path)

        models[product_code] = {
            "model": artifact["model"],
            "threshold": artifact["threshold"],
        }

    return models


# ----------------------------------------
# 3. Generate propensity scores
# ----------------------------------------

def add_propensity_scores(df, models):

    result = df.copy()

    X = result[FEATURE_COLUMNS].copy()

    for product_code, model_info in models.items():

        model = model_info["model"]

        result[f"{product_code}_propensity"] = (
            model.predict_proba(X)[:, 1]
        )

    return result


# ----------------------------------------
# 4. Apply business rules
# ----------------------------------------

def is_true(value):
    """Safely interpret common numeric and boolean flag values."""

    return str(value).strip().lower() in {
        "1", "1.0", "true", "yes"
    }


def make_decision(customer, models):

    customer_id = customer["customer_id"]

    delinquent = is_true(customer["delinquent_flag"])

    emi_card_blocked = is_true(
        customer["emi_card_blocked"]
    )

    low_engagement = (
        customer["app_sessions_30d"] < LOW_SESSION_LIMIT
        and customer["days_since_last_login"] > INACTIVE_DAYS_LIMIT
    )

    # EMI-card block affects EMI-card promotions only in this prototype.
    emi_promo_suppressed = emi_card_blocked

    # Rule 1: Delinquent customers do not receive credit recommendations.
    if delinquent:
        return {
            "customer_id": customer_id,
            "recommended_product": "No credit offer",
            "recommended_score": None,
            "decision_reason": "Suppressed: delinquency rule",
            "emi_promo_suppressed": emi_promo_suppressed,
        }

    # Rule 2: Low engagement customers receive re-engagement content.
    if low_engagement:
        return {
            "customer_id": customer_id,
            "recommended_product": "Re-engagement content",
            "recommended_score": None,
            "decision_reason": "Low engagement: show re-engagement content",
            "emi_promo_suppressed": emi_promo_suppressed,
        }

    candidates = []

    # Rule 3: Check eligibility and propensity for each product.
    for product_code, config in PRODUCTS.items():

        eligible = is_true(
            customer[config["eligibility_column"]]
        )

        propensity = float(
            customer[f"{product_code}_propensity"]
        )

        threshold = models[product_code]["threshold"]

        if eligible and propensity >= threshold:

            candidates.append({
                "product_code": product_code,
                "product_name": config["name"],
                "propensity": propensity,
            })

    # Rule 4: Rank qualifying products by propensity.
    if candidates:

        candidates.sort(
            key=lambda item: item["propensity"],
            reverse=True,
        )

        winner = candidates[0]

        return {
            "customer_id": customer_id,
            "recommended_product": winner["product_name"],
            "recommended_score": round(
                winner["propensity"], 4
            ),
            "decision_reason": (
                f"Eligible and above propensity threshold; "
                f"highest qualifying score among "
                f"{len(candidates)} candidate(s)"
            ),
            "emi_promo_suppressed": emi_promo_suppressed,
        }

    # Rule 5: No qualifying product.
    return {
        "customer_id": customer_id,
        "recommended_product": "General engagement content",
        "recommended_score": None,
        "decision_reason": (
            "No eligible product met its propensity threshold"
        ),
        "emi_promo_suppressed": emi_promo_suppressed,
    }


# ----------------------------------------
# 5. Build homepage recommendations
# ----------------------------------------

def main():

    print("Loading test customer data...")

    input_file = DATA_DIR / "test.parquet"

    if not input_file.exists():
        raise FileNotFoundError(
            "test.parquet not found. Run split_data.py first."
        )

    customers = pd.read_parquet(input_file)

    print(f"Customers loaded: {len(customers)}")

    print("\nLoading propensity models...")

    models = load_models()

    print("Generating propensity scores...")

    scored_customers = add_propensity_scores(
        customers,
        models,
    )

    print("Applying business rules...")

    decisions = []

    for _, customer in scored_customers.iterrows():

        decision = make_decision(customer, models)

        decisions.append(decision)

    recommendations = pd.DataFrame(decisions)

    # Add propensity scores for analysis and transparency.
    for product_code in PRODUCTS:
        recommendations[f"{product_code}_propensity"] = (
            scored_customers[
                f"{product_code}_propensity"
            ].values
        )

    output_file = (
        REPORT_DIR / "homepage_recommendations.csv"
    )

    recommendations.to_csv(
        output_file,
        index=False,
    )

    print("\nRecommendation distribution:")
    print(
        recommendations["recommended_product"]
        .value_counts()
        .to_string()
    )

    print("\nSample recommendations:")

    print(
        recommendations[
            [
                "customer_id",
                "recommended_product",
                "recommended_score",
                "decision_reason",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nSaved recommendations:", output_file)


if __name__ == "__main__":
    main()