
from pathlib import Path

import pandas as pd

from decision_engine import load_models, add_propensity_scores, make_decision, PRODUCTS


# =========================================================
# 1. LOAD TEST DATA AND MODELS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "test.parquet"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Run split_data.py first."
    )

customers = pd.read_parquet(DATA_FILE)
models = load_models()

customers = add_propensity_scores(customers, models)


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def is_true(value):
    return str(value).strip().lower() in {
        "1", "1.0", "true", "yes"
    }


def is_low_engagement(customer):
    sessions = float(customer["app_sessions_30d"])
    days_since_login = float(customer["days_since_last_login"])

    return sessions < 2 and days_since_login > 14


# Map the displayed recommendation back to its model.
PRODUCT_NAME_TO_CODE = {
    config["name"]: code
    for code, config in PRODUCTS.items()
}


# =========================================================
# 3. RUN RECOMMENDATION TESTS
# =========================================================

results = {
    "total_customers": len(customers),
    "delinquency_checks": 0,
    "low_engagement_checks": 0,
    "eligibility_checks": 0,
    "threshold_checks": 0,
    "emi_block_checks": 0,
}

failures = []

recommendation_counts = {}

for _, customer in customers.iterrows():

    decision = make_decision(customer, models)

    customer_id = customer["customer_id"]
    recommendation = decision["recommended_product"]

    recommendation_counts[recommendation] = (
        recommendation_counts.get(recommendation, 0) + 1
    )

    # -----------------------------------------------------
    # TEST 1: DELINQUENCY SUPPRESSION
    # -----------------------------------------------------

    if is_true(customer["delinquent_flag"]):

        results["delinquency_checks"] += 1

        if recommendation != "No credit offer":
            failures.append({
                "customer_id": customer_id,
                "test": "Delinquency suppression",
                "expected": "No credit offer",
                "actual": recommendation,
            })

    # -----------------------------------------------------
    # TEST 2: LOW-ENGAGEMENT RULE
    # -----------------------------------------------------

    elif is_low_engagement(customer):

        results["low_engagement_checks"] += 1

        if recommendation != "Re-engagement content":
            failures.append({
                "customer_id": customer_id,
                "test": "Low engagement",
                "expected": "Re-engagement content",
                "actual": recommendation,
            })

    # -----------------------------------------------------
    # TEST 3: PRODUCT ELIGIBILITY
    # -----------------------------------------------------

    if recommendation in PRODUCT_NAME_TO_CODE:

        results["eligibility_checks"] += 1

        product_code = PRODUCT_NAME_TO_CODE[recommendation]
        eligibility_column = PRODUCTS[product_code][
            "eligibility_column"
        ]

        if not is_true(customer[eligibility_column]):
            failures.append({
                "customer_id": customer_id,
                "test": "Product eligibility",
                "expected": f"{product_code} eligible",
                "actual": "Recommended while ineligible",
            })

        # -------------------------------------------------
        # TEST 4: MODEL THRESHOLD
        # -------------------------------------------------

        results["threshold_checks"] += 1

        score_column = f"{product_code}_propensity"
        score = float(customer[score_column])
        threshold = float(models[product_code]["threshold"])

        if score < threshold:
            failures.append({
                "customer_id": customer_id,
                "test": "Propensity threshold",
                "expected": f"Score >= {threshold:.3f}",
                "actual": f"Score = {score:.3f}",
            })

    # -----------------------------------------------------
    # TEST 5: EMI PROMOTIONAL SUPPRESSION
    # -----------------------------------------------------

    if is_true(customer["emi_card_blocked"]):

        results["emi_block_checks"] += 1

        if not decision["emi_promo_suppressed"]:
            failures.append({
                "customer_id": customer_id,
                "test": "EMI promotional suppression",
                "expected": "EMI promotion suppressed",
                "actual": "EMI promotion not suppressed",
            })


# =========================================================
# 4. DISPLAY TEST RESULTS
# =========================================================

print("\n" + "=" * 65)
print("       RECOMMENDATION ENGINE VALIDATION REPORT")
print("=" * 65)

print(f"\nTotal customers tested: {results['total_customers']:,}")

print("\nTEST COVERAGE")
print("-" * 65)

for name, count in results.items():
    if name != "total_customers":
        print(f"{name.replace('_', ' ').title():35} {count:>6,}")

print("\nRECOMMENDATION DISTRIBUTION")
print("-" * 65)

for recommendation, count in sorted(
    recommendation_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{recommendation:35} {count:>6,}")

print("\nVALIDATION SUMMARY")
print("-" * 65)

if failures:
    print(f"FAILED: {len(failures)} rule violations found.")

    failure_df = pd.DataFrame(failures)

    print("\nFirst 20 failures:")
    print(failure_df.head(20).to_string(index=False))

    report_file = ROOT / "reports" / "recommendation_test_failures.csv"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    failure_df.to_csv(report_file, index=False)

    print(f"\nFailure report saved to: {report_file}")

else:
    print("PASSED: No rule violations found.")

print("\n" + "=" * 65)
