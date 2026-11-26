
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from decision_engine import load_models
from feature_config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


# =========================================================
# 1. PROJECT PATHS AND SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "test.parquet"
OUTPUT_DIR = ROOT / "reports" / "shap"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EXPLANATION_ROWS = 500
RANDOM_STATE = 42

PRODUCTS = {
    "pl": "Personal Loan",
    "sme": "SME Loan",
    "b2b": "B2B Finance",
}


# =========================================================
# 2. LOAD TEST DATA AND MODELS
# =========================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {DATA_FILE}"
    )

customers = pd.read_parquet(DATA_FILE)

models = load_models()

feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

missing_columns = [
    column for column in feature_columns
    if column not in customers.columns
]

if missing_columns:
    raise ValueError(
        f"Missing model input columns: {missing_columns}"
    )

X = customers[feature_columns].copy()

# Use a reproducible sample to keep SHAP computation manageable.
X_sample = X.sample(
    n=min(MAX_EXPLANATION_ROWS, len(X)),
    random_state=RANDOM_STATE
)

print("=" * 70)
print("             SHAP MODEL EXPLAINABILITY")
print("=" * 70)

print(f"Test customers: {len(customers):,}")
print(f"Customers explained per model: {len(X_sample):,}")


# =========================================================
# 3. HELPER: EXTRACT TRANSFORMED FEATURES
# =========================================================

def transform_features(pipeline, raw_data):

    if not hasattr(pipeline, "steps"):
        raise TypeError(
            "Expected a scikit-learn Pipeline in the model artifact. "
            "Please share the model artifact structure."
        )

    if len(pipeline.steps) < 2:
        raise ValueError(
            "Expected a preprocessing step followed by a model step."
        )

    preprocessing_pipeline = pipeline[:-1]
    estimator = pipeline.steps[-1][1]

    transformed = preprocessing_pipeline.transform(raw_data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    try:
        feature_names = preprocessing_pipeline.get_feature_names_out()
        feature_names = [str(name) for name in feature_names]
    except Exception:
        feature_names = [
            f"feature_{index}"
            for index in range(transformed.shape[1])
        ]

    if len(feature_names) != transformed.shape[1]:
        feature_names = [
            f"feature_{index}"
            for index in range(transformed.shape[1])
        ]

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=raw_data.index
    )

    return estimator, transformed_df


# =========================================================
# 4. HELPER: NORMALIZE SHAP OUTPUT
# =========================================================

def get_positive_class_shap_values(explainer, transformed_df):

    shap_values = explainer.shap_values(transformed_df)

    # Older SHAP versions may return one array per class.
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    # Some versions return:
    # rows x features x classes
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    if shap_values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    return shap_values


# =========================================================
# 5. EXPLAIN EACH PRODUCT MODEL
# =========================================================

for product_code, product_name in PRODUCTS.items():

    print("\n" + "=" * 70)
    print(f"EXPLAINING MODEL: {product_name.upper()}")
    print("=" * 70)

    if product_code not in models:
        print(f"Model {product_code} not found. Skipping.")
        continue

    pipeline = models[product_code]["model"]

    # Transform raw customer features using the fitted pipeline.
    estimator, transformed_df = transform_features(
        pipeline,
        X_sample
    )

    # Explain the XGBoost estimator on transformed features.
    explainer = shap.TreeExplainer(estimator)

    shap_values = get_positive_class_shap_values(
        explainer,
        transformed_df
    )

    # -----------------------------------------------------
    # 5A. GLOBAL FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": transformed_df.columns,
        "mean_absolute_shap": np.abs(shap_values).mean(axis=0)
    })

    importance_df = importance_df.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    importance_file = (
        OUTPUT_DIR / f"{product_code}_feature_importance.csv"
    )

    importance_df.to_csv(importance_file, index=False)

    print("\nTop 10 influential features:")

    print(
        importance_df.head(10).to_string(index=False)
    )

    # -----------------------------------------------------
    # 5B. SHAP SUMMARY PLOT
    # -----------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        transformed_df,
        max_display=15,
        show=False
    )

    plt.title(f"{product_name} - SHAP Summary")

    plt.tight_layout()

    summary_file = OUTPUT_DIR / f"{product_code}_shap_summary.png"

    plt.savefig(
        summary_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSummary plot saved: {summary_file.name}")

    # -----------------------------------------------------
    # 5C. GLOBAL FEATURE IMPORTANCE BAR PLOT
    # -----------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        transformed_df,
        plot_type="bar",
        max_display=15,
        show=False
    )

    plt.title(f"{product_name} - Feature Importance")

    plt.tight_layout()

    bar_file = OUTPUT_DIR / f"{product_code}_shap_bar.png"

    plt.savefig(
        bar_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Bar plot saved: {bar_file.name}")

    # -----------------------------------------------------
    # 5D. INDIVIDUAL CUSTOMER EXPLANATION
    # -----------------------------------------------------

    # Explain the first customer in the sampled test data.
    customer_index = 0

    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).reshape(-1)[-1]

    explanation = shap.Explanation(
        values=shap_values[customer_index],
        base_values=float(expected_value),
        data=transformed_df.iloc[customer_index].values,
        feature_names=transformed_df.columns.tolist()
    )

    plt.figure()

    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False
    )

    plt.title(
        f"{product_name} - Individual Customer Explanation"
    )

    plt.tight_layout()

    waterfall_file = (
        OUTPUT_DIR / f"{product_code}_customer_waterfall.png"
    )

    plt.savefig(
        waterfall_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Individual explanation saved: {waterfall_file.name}")


# =========================================================
# 6. FINAL OUTPUT
# =========================================================

print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY COMPLETED")
print("=" * 70)

print(f"Reports saved in: {OUTPUT_DIR}")

print("\nGenerated files include:")
print("- Feature importance CSV files")
print("- SHAP summary plots")
print("- SHAP feature-importance bar plots")
print("- Individual customer waterfall plots")

print("=" * 70)
