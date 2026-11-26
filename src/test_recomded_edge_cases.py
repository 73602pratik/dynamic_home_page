
from pathlib import Path

import pandas as pd

from decision_engine import load_models, add_propensity_scores, make_decision, PRODUCTS


# =========================================================
# 1. LOAD DATA AND MODELS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "test.parquet"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "test.parquet not found. Run split_data.py first."
    )

customers = pd.read_parquet(DATA_FILE)
models = load_models()

customers = add_propensity_scores(customers, models)

# Use one customer as a template, then create controlled scenarios.
base_customer = customers.iloc[0].copy()


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def set_all_eligibility(customer, eligible=False):
    for product_code, config in PRODUCTS.items():
        column = config["eligibility_column"]
        customer[column] = eligible


def set_scores_above_threshold(customer, product_code, score=0.99):
    customer[f"{product_code}_propensity"] = score


def run_test(test_name, customer, expected_recommendation=None,
             expected_emi_suppressed=None):

    decision = make_decision(customer, models)

    actual_recommendation = decision["recommended_product"]
    actual_emi_suppressed = decision["emi_promo_suppressed"]

    passed = True
    reasons = []

    if (
        expected_recommendation is not None
        and actual_recommendation != expected_recommendation
    ):
        passed = False
        reasons.append(
            f"Expected recommendation: {expected_recommendation}; "
            f"Actual: {actual_recommendation}"
        )

    if (
        expected_emi_suppressed is not None
        and actual_emi_suppressed != expected_emi_suppressed
    ):
        passed = False
        reasons.append(
            f"Expected EMI suppression: {expected_emi_suppressed}; "
            f"Actual: {actual_emi_suppressed}"
        )

    print(f"\n{test_name}")
    print(f"Recommendation: {actual_recommendation}")
    print(f"EMI promotion suppressed: {actual_emi_suppressed}")

    if passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        for reason in reasons:
            print(f"  - {reason}")

    return {
        "test": test_name,
        "expected": expected_recommendation,
        "actual": actual_recommendation,
        "emi_suppressed": actual_emi_suppressed,
        "result": "PASS" if passed else "FAIL",
        "details": "; ".join(reasons),
    }


test_results = []


# =========================================================
# 3. TEST 1 — DELINQUENT CUSTOMER
# =========================================================

customer = base_customer.copy()

set_all_eligibility(customer, eligible=True)
customer["delinquent_flag"] = 1
customer["app_sessions_30d"] = 10
customer["days_since_last_login"] = 2

for code in PRODUCTS:
    set_scores_above_threshold(customer, code, 0.99)

test_results.append(
    run_test(
        "Test 1: Delinquent customer",
        customer,
        expected_recommendation="No credit offer",
    )
)


# =========================================================
# 4. TEST 2 — LOW-ENGAGEMENT CUSTOMER
# =========================================================

customer = base_customer.copy()

set_all_eligibility(customer, eligible=True)
customer["delinquent_flag"] = 0
customer["app_sessions_30d"] = 0
customer["days_since_last_login"] = 30

for code in PRODUCTS:
    set_scores_above_threshold(customer, code, 0.99)

test_results.append(
    run_test(
        "Test 2: Low-engagement customer",
        customer,
        expected_recommendation="Re-engagement content",
    )
)


# =========================================================
# 5. TEST 3 — NO ELIGIBLE PRODUCTS
# =========================================================

customer = base_customer.copy()

set_all_eligibility(customer, eligible=False)
customer["delinquent_flag"] = 0
customer["app_sessions_30d"] = 10
customer["days_since_last_login"] = 2

test_results.append(
    run_test(
        "Test 3: No eligible products",
        customer,
        expected_recommendation="General engagement content",
    )
)


# =========================================================
# 6. TEST 4 — MULTIPLE ELIGIBLE PRODUCTS
# =========================================================

customer = base_customer.copy()

set_all_eligibility(customer, eligible=False)
customer["delinquent_flag"] = 0
customer["app_sessions_30d"] = 10
customer["days_since_last_login"] = 2

# Make PL and SME eligible, but not B2B.
customer[PRODUCTS["pl"]["eligibility_column"]] = 1
customer[PRODUCTS["sme"]["eligibility_column"]] = 1

# Both scores exceed their model thresholds.
# PL is deliberately given the highest score.
customer["pl_propensity"] = 0.999999
customer["sme_propensity"] = (
    models["sme"]["threshold"] + 0.01
)

customer["b2b_propensity"] = 0.0

test_results.append(
    run_test(
        "Test 4: Multiple eligible products",
        customer,
        expected_recommendation=PRODUCTS["pl"]["name"],
    )
)


# =========================================================
# 7. TEST 5 — EMI BLOCKED BUT LOAN ELIGIBLE
# =========================================================

customer = base_customer.copy()

set_all_eligibility(customer, eligible=False)
customer["delinquent_flag"] = 0
customer["app_sessions_30d"] = 10
customer["days_since_last_login"] = 2

customer[PRODUCTS["pl"]["eligibility_column"]] = 1
customer["pl_propensity"] = 0.999999

customer["emi_card_blocked"] = 1

test_results.append(
    run_test(
        "Test 5: EMI blocked but Personal Loan eligible",
        customer,
        expected_recommendation=PRODUCTS["pl"]["name"],
        expected_emi_suppressed=True,
    )
)


# =========================================================
# 8. SAVE AND DISPLAY TEST REPORT
# =========================================================

results_df = pd.DataFrame(test_results)

report_file = ROOT / "reports" / "recommendation_edge_case_results.csv"
report_file.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(report_file, index=False)

passed = (results_df["result"] == "PASS").sum()
failed = (results_df["result"] == "FAIL").sum()

print("\n" + "=" * 65)
print("       EDGE-CASE VALIDATION SUMMARY")
print("=" * 65)

print(f"Total scenarios: {len(results_df)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed == 0:
    print("\nALL EDGE-CASE TESTS PASSED.")
else:
    print("\nSOME EDGE-CASE TESTS FAILED.")
    print(
        results_df.loc[
            results_df["result"] == "FAIL"
        ].to_string(index=False)
    )

print(f"\nReport saved to: {report_file}")
print("=" * 65)
