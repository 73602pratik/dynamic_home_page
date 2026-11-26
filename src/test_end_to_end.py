
from pathlib import Path

import pandas as pd

from decision_engine import load_models, add_propensity_scores, make_decision
from homepage_personalizer import personalize_customer


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

print("=" * 70)
print("       END-TO-END PERSONALIZATION PIPELINE TEST")
print("=" * 70)

print(f"\nCustomers loaded: {len(customers):,}")


# =========================================================
# 2. INITIALIZE TEST RESULTS
# =========================================================

checks = {
    "recommendation_consistency": 0,
    "score_consistency": 0,
    "decision_reason_consistency": 0,
    "emi_suppression_consistency": 0,
    "homepage_fields_present": 0,
}

failures = []
homepage_records = []


# =========================================================
# 3. TEST EACH CUSTOMER THROUGH THE PIPELINE
# =========================================================

for _, customer in customers.iterrows():

    customer_id = customer["customer_id"]

    # Decision engine output
    decision = make_decision(customer, models)

    # Homepage personalization output
    homepage = personalize_customer(customer, models)

    # -----------------------------------------------------
    # CHECK 1: RECOMMENDATION CONSISTENCY
    # -----------------------------------------------------

    checks["recommendation_consistency"] += 1

    if (
        homepage.get("recommended_product")
        != decision.get("recommended_product")
    ):
        failures.append({
            "customer_id": customer_id,
            "test": "Recommendation consistency",
            "decision_engine": decision.get("recommended_product"),
            "homepage": homepage.get("recommended_product"),
        })

    # -----------------------------------------------------
    # CHECK 2: PROPENSITY SCORE CONSISTENCY
    # -----------------------------------------------------

    checks["score_consistency"] += 1

    homepage_score = homepage.get("recommended_score")
    decision_score = decision.get("recommended_score")

    scores_match = (
        homepage_score is None and decision_score is None
    )

    if homepage_score is not None and decision_score is not None:
        try:
            scores_match = abs(
                float(homepage_score) - float(decision_score)
            ) < 1e-9
        except (TypeError, ValueError):
            scores_match = False

    if not scores_match:
        failures.append({
            "customer_id": customer_id,
            "test": "Recommended score consistency",
            "decision_engine": decision_score,
            "homepage": homepage_score,
        })

    # -----------------------------------------------------
    # CHECK 3: DECISION REASON CONSISTENCY
    # -----------------------------------------------------

    checks["decision_reason_consistency"] += 1

    if (
        homepage.get("decision_reason")
        != decision.get("decision_reason")
    ):
        failures.append({
            "customer_id": customer_id,
            "test": "Decision reason consistency",
            "decision_engine": decision.get("decision_reason"),
            "homepage": homepage.get("decision_reason"),
        })

    # -----------------------------------------------------
    # CHECK 4: EMI PROMOTIONAL SUPPRESSION
    # -----------------------------------------------------

    checks["emi_suppression_consistency"] += 1

    if (
        homepage.get("emi_promo_suppressed")
        != decision.get("emi_promo_suppressed")
    ):
        failures.append({
            "customer_id": customer_id,
            "test": "EMI suppression consistency",
            "decision_engine": decision.get("emi_promo_suppressed"),
            "homepage": homepage.get("emi_promo_suppressed"),
        })

    # -----------------------------------------------------
    # CHECK 5: REQUIRED HOMEPAGE FIELDS
    # -----------------------------------------------------

    checks["homepage_fields_present"] += 1

    required_fields = [
        "customer_id",
        "cohort",
        "recommended_product",
        "primary_slot",
        "secondary_slot",
        "banner",
        "decision_reason",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in homepage or homepage[field] is None
    ]

    if missing_fields:
        failures.append({
            "customer_id": customer_id,
            "test": "Homepage fields",
            "decision_engine": "Not applicable",
            "homepage": f"Missing fields: {missing_fields}",
        })

    # Save homepage result for reporting
    homepage_records.append(homepage)


# =========================================================
# 4. CREATE INTEGRATION REPORT
# =========================================================

homepage_df = pd.DataFrame(homepage_records)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

homepage_report = REPORT_DIR / "end_to_end_homepage_results.csv"

homepage_df.to_csv(homepage_report, index=False)


# =========================================================
# 5. DISPLAY TEST COVERAGE
# =========================================================

print("\nINTEGRATION TEST COVERAGE")
print("-" * 70)

for check_name, count in checks.items():
    print(f"{check_name.replace('_', ' ').title():40} {count:>7,}")


# =========================================================
# 6. DISPLAY RECOMMENDATION DISTRIBUTION
# =========================================================

print("\nHOMEPAGE RECOMMENDATION DISTRIBUTION")
print("-" * 70)

if "recommended_product" in homepage_df.columns:
    distribution = homepage_df["recommended_product"].value_counts()

    for product, count in distribution.items():
        print(f"{product:40} {count:>7,}")


# =========================================================
# 7. DISPLAY COHORT DISTRIBUTION
# =========================================================

print("\nCUSTOMER COHORT DISTRIBUTION")
print("-" * 70)

if "cohort" in homepage_df.columns:
    cohort_distribution = homepage_df["cohort"].value_counts()

    for cohort, count in cohort_distribution.items():
        print(f"{cohort:40} {count:>7,}")


# =========================================================
# 8. FINAL VALIDATION
# =========================================================

print("\n" + "=" * 70)
print("       END-TO-END VALIDATION SUMMARY")
print("=" * 70)

total_checks = sum(checks.values())
failed_checks = len(failures)
passed_checks = total_checks - failed_checks

print(f"Customers processed: {len(customers):,}")
print(f"Checks executed:      {total_checks:,}")
print(f"Checks passed:        {passed_checks:,}")
print(f"Failures found:       {failed_checks:,}")

if failures:

    print("\nFAILED CHECKS")
    print("-" * 70)

    failure_df = pd.DataFrame(failures)
    print(failure_df.head(20).to_string(index=False))

    failure_report = REPORT_DIR / "end_to_end_failures.csv"
    failure_df.to_csv(failure_report, index=False)

    print(f"\nFailure report saved to: {failure_report}")

else:
    print("\nSUCCESS: ALL END-TO-END CHECKS PASSED.")

print(f"\nHomepage output saved to: {homepage_report}")

print("=" * 70)
