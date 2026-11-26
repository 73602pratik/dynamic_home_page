
import numpy as np
import pandas as pd
import shap
import streamlit as st

from feature_config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from decision_engine import load_models


FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_resource
def load_shap_explainers():
    """Load and cache one SHAP TreeExplainer for each trained model."""
    models = load_models()
    explainers = {}

    for product_code, model_info in models.items():
        pipeline = model_info["model"]
        estimator = pipeline.steps[-1][1]
        explainers[product_code] = shap.TreeExplainer(estimator)

    return explainers


def explain_customer(customer, product_code, models, explainers, top_n=5):
    """
    Explain one customer's prediction for one product.

    SHAP contributions are in the model's raw output scale,
    not percentage points of propensity.
    """
    if product_code not in models or product_code not in explainers:
        raise ValueError(f"Unknown product code: {product_code}")

    pipeline = models[product_code]["model"]

    # Keep only the original features used during model training.
    input_data = pd.DataFrame(
        [{column: customer[column] for column in FEATURE_COLUMNS}]
    )

    # Apply the exact preprocessing used by the trained model.
    preprocessor = pipeline[:-1]
    transformed = preprocessor.transform(input_data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    explainer = explainers[product_code]

    # Calculate SHAP values for this one customer.
    values = explainer.shap_values(transformed)

    # Handle SHAP's different output formats for binary classifiers.
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    values = np.asarray(values)

    if values.ndim == 3:
        # Select the positive class for a binary classifier.
        values = values[:, :, 1]

    if values.ndim == 1:
        values = values.reshape(1, -1)

    contributions = values[0]
    transformed_values = np.asarray(transformed)[0]

    if len(feature_names) != len(contributions):
        raise ValueError("Feature names and SHAP values have different lengths.")

    result = pd.DataFrame({
        "Feature": [
            name.replace("numeric__", "").replace("categorical__", "")
            for name in feature_names
        ],
        "Model input value": transformed_values,
        "SHAP contribution": contributions
    })

    result["Effect"] = np.where(
        result["SHAP contribution"] >= 0,
        "Pushes score higher",
        "Pushes score lower"
    )

    result["Absolute contribution"] = result["SHAP contribution"].abs()

    result = result.sort_values(
        "Absolute contribution",
        ascending=False
    ).head(top_n)

    return result[
        ["Feature", "Model input value", "SHAP contribution", "Effect"]
    ].reset_index(drop=True)
