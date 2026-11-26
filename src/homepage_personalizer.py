
from pathlib import Path

import pandas as pd

from decision_engine import (
    load_models,
    add_propensity_scores,
    make_decision,
    PRODUCTS,
)

from feature_config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


# ============================================
# 1. PROJECT CONFIGURATION
# ============================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)


# ============================================
# 2. HELPER FUNCTIONS
# ============================================

def is_true(value):

    return str(value).strip().lower() in {
        "1", "1.0", "true", "yes"
    }


# ============================================
# 3. FIND QUALIFYING PRODUCTS
# ============================================

def get_product_candidates(customer, models):

    candidates = []

    for product_code, config in PRODUCTS.items():

        eligible = is_true(
            customer[config["eligibility_column"]]
        )

        score = float(
            customer[f"{product_code}_propensity"]
        )

        threshold = models[product_code]["threshold"]

        if eligible and score >= threshold:

            candidates.append({
                "code": product_code,
                "name": config["name"],
                "score": score,
            })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates


# ============================================
# 4. ASSIGN COHORT FROM CENTRAL DECISION
# ============================================

def assign_cohort(decision):

    recommended_product = decision["recommended_product"]

    cohort_mapping = {
        "Personal Loan": "PL",
        "SME Loan": "SME",
        "B2B Finance": "B2B",
        "No credit offer": "DMS",
        "Re-engagement content": "Engagement",
        "General engagement content": "Other",
    }

    return cohort_mapping.get(
        recommended_product,
        "Other",
    )


# ============================================
# 5. CREATE HOMEPAGE CONTENT SLOTS
# ============================================

def create_homepage_slots(
    customer,
    candidates,
    decision,
):

    recommended_product = decision["recommended_product"]

    emi_blocked = is_true(
        customer["emi_card_blocked"]
    )

    # DMS customer
    if recommended_product == "No credit offer":

        return {
            "primary_slot": (
                "Account support and repayment assistance"
            ),
            "secondary_slot": (
                "Financial education content"
            ),
            "banner": "Account support information",
        }

    # Low engagement customer
    if recommended_product == "Re-engagement content":

        return {
            "primary_slot": "Re-engagement content",
            "secondary_slot": "Explore available products",
            "banner": "Welcome back - discover what's new",
        }

    # No qualifying product
    if recommended_product == "General engagement content":

        return {
            "primary_slot": "General engagement content",
            "secondary_slot": "Explore financial products",
            "banner": "Explore products and services",
        }

    # Product recommendation
    primary = recommended_product

    # Exclude primary product from secondary candidates.
    secondary_candidates = [
        candidate
        for candidate in candidates
        if candidate["name"] != primary
    ]

    if secondary_candidates:

        secondary = secondary_candidates[0]["name"]

    else:

        secondary = "General engagement content"

    banner = f"Explore {primary}"

    if emi_blocked:

        banner = "Explore available financial products"

    return {
        "primary_slot": primary,
        "secondary_slot": secondary,
        "banner": banner,
    }


# ============================================
# 6. UNIFIED PERSONALIZATION PIPELINE
# ============================================

def personalize_customer(customer, models):

    # Step 1: Central business-rule decision
    decision = make_decision(
        customer,
        models,
    )

    # Step 2: Determine qualifying products
    candidates = get_product_candidates(
        customer,
        models,
    )

    # Step 3: Assign cohort using central decision
    cohort = assign_cohort(decision)

    # Step 4: Create homepage slots
    slots = create_homepage_slots(
        customer,
        candidates,
        decision,
    )

    # Step 5: Build final customer experience
    result = {

        "customer_id": customer["customer_id"],

        "cohort": cohort,

        "recommended_product": (
            decision["recommended_product"]
        ),

        "recommended_score": (
            decision["recommended_score"]
        ),

        "pl_propensity": float(
            customer["pl_propensity"]
        ),

        "sme_propensity": float(
            customer["sme_propensity"]
        ),

        "b2b_propensity": float(
            customer["b2b_propensity"]
        ),

        "primary_slot": slots["primary_slot"],

        "secondary_slot": slots["secondary_slot"],

        "banner": slots["banner"],

        "decision_reason": decision["decision_reason"],

        "emi_promo_suppressed": (
            decision["emi_promo_suppressed"]
        ),

        "delinquent_flag": customer["delinquent_flag"],

        "emi_card_blocked": customer["emi_card_blocked"],

        "candidate_count": len(candidates),

    }

    return result


# ============================================
# 7. MAIN EXECUTION
# ============================================

def main():

    print("Loading customer data...")

    input_file = DATA_DIR / "test.parquet"

    if not input_file.exists():

        raise FileNotFoundError(
            "test.parquet not found. Run split_data.py first."
        )

    customers = pd.read_parquet(input_file)

    print(f"Customers loaded: {len(customers)}")

    print("\nLoading models...")

    models = load_models()

    print("\nGenerating propensity scores...")

    scored_customers = add_propensity_scores(
        customers,
        models,
    )

    required_columns = [
        "customer_id",
        "pl_propensity",
        "sme_propensity",
        "b2b_propensity",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in scored_customers.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("\nCreating unified personalized homepages...")

    results = []

    for _, customer in scored_customers.iterrows():

        result = personalize_customer(
            customer,
            models,
        )

        results.append(result)

    homepage_df = pd.DataFrame(results)

    # Save output
    output_file = (
        REPORT_DIR / "personalized_homepages.csv"
    )

    homepage_df.to_csv(
        output_file,
        index=False,
    )

    # Cohort summary
    print("\nCustomer cohort distribution:")

    print(
        homepage_df["cohort"]
        .value_counts()
        .to_string()
    )

    # Product recommendation summary
    print("\nRecommended product distribution:")

    print(
        homepage_df["recommended_product"]
        .value_counts()
        .to_string()
    )

    # Homepage slots
    print("\nPrimary homepage slot distribution:")

    print(
        homepage_df["primary_slot"]
        .value_counts()
        .to_string()
    )

    print("\nSample personalized homepages:")

    print(
        homepage_df[
            [
                "customer_id",
                "cohort",
                "recommended_product",
                "primary_slot",
                "secondary_slot",
                "banner",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTotal personalized customers:", len(homepage_df))

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
