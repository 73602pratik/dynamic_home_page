from pathlib import Path
import json

import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from feature_config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGETS,
)


# ----------------------------------------
# 1. Paths
# ----------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ----------------------------------------
# 2. Load datasets
# ----------------------------------------

def load_datasets():

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    return train, validation, test


# ----------------------------------------
# 3. Prepare input features
# ----------------------------------------

def prepare_features(df, target):

    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[columns].copy()
    y = df[target].astype(int)

    return X, y


# ----------------------------------------
# 4. Build model
# ----------------------------------------

def build_pipeline(scale_pos_weight):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ----------------------------------------
# 5. Find best threshold using validation
# ----------------------------------------

def find_best_threshold(y_true, probabilities):

    best_threshold = 0.5
    best_f1 = -1

    for threshold in [x / 100 for x in range(5, 96, 5)]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


# ----------------------------------------
# 6. Evaluate model
# ----------------------------------------

def evaluate_model(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),

        "average_precision": float(
            average_precision_score(
                y_true, probabilities
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


# ----------------------------------------
# 7. Train one propensity model
# ----------------------------------------

def train_one_model(
    model_name,
    target,
    train_df,
    validation_df,
    test_df
):

    print("\n" + "=" * 50)
    print(f"Training {model_name.upper()} model")
    print("=" * 50)

    X_train, y_train = prepare_features(train_df, target)
    X_val, y_val = prepare_features(validation_df, target)
    X_test, y_test = prepare_features(test_df, target)

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)

    if positive_count == 0:
        raise ValueError(
            f"No positive training examples for {target}"
        )

    scale_pos_weight = negative_count / positive_count

    print(f"Positive training records: {positive_count}")
    print(f"Negative training records: {negative_count}")
    print(f"Scale positive weight: {scale_pos_weight:.2f}")

    model = build_pipeline(scale_pos_weight)

    print("Fitting model...")

    model.fit(X_train, y_train)

    # Select threshold using validation data
    validation_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    threshold = find_best_threshold(
        y_val,
        validation_probabilities
    )

    print(f"Selected threshold: {threshold:.2f}")

    # Final test evaluation
    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = evaluate_model(
        y_test,
        test_probabilities,
        threshold
    )

    print("\nTest metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    # Save model and its threshold
    model_path = MODEL_DIR / f"{model_name}_propensity_model.joblib"

    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "target": target,
            "model_name": model_name,
        },
        model_path
    )

    metrics_path = REPORT_DIR / f"{model_name}_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved model: {model_path}")

    return metrics


# ----------------------------------------
# 8. Train all three models
# ----------------------------------------

def main():

    train_df, validation_df, test_df = load_datasets()

    all_results = {}

    for model_name, target in TARGETS.items():

        metrics = train_one_model(
            model_name,
            target,
            train_df,
            validation_df,
            test_df
        )

        all_results[model_name] = metrics

    # Save combined summary
    summary_path = REPORT_DIR / "all_models_summary.json"

    with open(summary_path, "w") as file:
        json.dump(all_results, file, indent=4)

    print("\nAll three models trained successfully.")
    print("Combined summary:", summary_path)


if __name__ == "__main__":
    main()