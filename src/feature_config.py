# Numeric predictors
NUMERIC_FEATURES = [
    "age",
    "monthly_income",
    "tenure_months",
    "app_sessions_30d",
    "product_views_30d",
    "loan_page_visits_30d",
    "days_since_last_login",
    "existing_products",
    "delinquent_flag",
    "emi_card_blocked",
    "engagement_score",]
# Categorical predictors
CATEGORICAL_FEATURES = [
    "customer_type",
    "income_band",]
# Target labels: one model per product
TARGETS = {
    "pl": "applied_pl_90d",
    "sme": "applied_sme_90d",
    "b2b": "applied_b2b_90d",}
# These columns must not be model predictors
EXCLUDED_FROM_PREDICTORS = [
    "customer_id",
    "eligible_pl",
    "eligible_sme",
    "eligible_b2b",
    "applied_pl_90d",
    "applied_sme_90d",
    "applied_b2b_90d",]
