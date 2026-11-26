from pathlib import Path
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,)
from feature_config import (NUMERIC_FEATURES,CATEGORICAL_FEATURES,TARGETS,)

# ------------------------------------------------
# 1. Project paths and configuration
# ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TARGET = TARGETS["pl"]

TRAIN_FILE = DATA_DIR / "train.parquet"
VALIDATION_FILE = DATA_DIR / "validation.parquet"
TEST_FILE = DATA_DIR / "test.parquet"

MODEL_FILE = MODEL_DIR / "pl_propensity_model.joblib"
METRICS_FILE = REPORT_DIR / "pl_model_metrics.json"

# ------------------------------------------------
# 2. Load datasets
# ------------------------------------------------

def load_data():
    train_df = pd.read_parquet(TRAIN_FILE)
    validation_df = pd.read_parquet(VALIDATION_FILE)
    test_df = pd.read_parquet(TEST_FILE)
    return train_df, validation_df, test_df

# ------------------------------------------------
# 3. Prepare features and target
# ------------------------------------------------

def prepare_xy(df):
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_columns = [
        col for col in feature_columns + [TARGET]
        if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    X = df[feature_columns].copy()
    y = df[TARGET].astype(int)
    return X, y

# ------------------------------------------------
# 4. Build preprocessing and model pipeline
# ------------------------------------------------

def build_model():
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),])

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent"),),
            ("onehot",OneHotEncoder(handle_unknown="ignore"),),])
    preprocessor = ColumnTransformer(
        transformers=[("numeric",numeric_pipeline,NUMERIC_FEATURES,),
            ("categorical",categorical_pipeline,CATEGORICAL_FEATURES,),])
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor),("classifier", xgb_model),])
    return pipeline

# ------------------------------------------------
# 5. Select classification threshold on validation
# ------------------------------------------------

def select_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_f1 = -1
    for threshold in [x / 100 for x in range(10, 91, 5)]:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true,predictions,zero_division=0,)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold

# ------------------------------------------------
# 6. Evaluate model
# ------------------------------------------------

def evaluate_model(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    results = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities )),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),}
    return results

# ------------------------------------------------
# 7. Main training workflow
# ------------------------------------------------

def main():

    print("Loading datasets...")
    train_df, validation_df, test_df = load_data()
    X_train, y_train = prepare_xy(train_df)
    X_validation, y_validation = prepare_xy(validation_df)
    X_test, y_test = prepare_xy(test_df)

    print(f"Training records: {len(X_train)}")
    print(f"Validation records: {len(X_validation)}")
    print(f"Testing records: {len(X_test)}")

    print("\nTraining Personal Loan propensity model...")

    model = build_model()

    model.fit(X_train, y_train)

    print("Model training completed.")

    # Validation: select the classification threshold
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    best_threshold = select_threshold(y_validation,validation_probabilities,)
    print(f"\nSelected validation threshold: "f"{best_threshold:.2f}")

    # Final evaluation: test set is used only here
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test,test_probabilities,best_threshold,)

    print("\nFinal test results")
    print("-" * 40)

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    # Save trained model
    joblib.dump(model, MODEL_FILE)

    # Save metrics
    with open(METRICS_FILE, "w") as file:
        json.dump(test_metrics, file, indent=4)

    print("\nModel saved:", MODEL_FILE)
    print("Metrics saved:", METRICS_FILE)


if __name__ == "__main__":
    main()