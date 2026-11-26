from pathlib import Path
import numpy as np
import pandas as pd

# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
INPUT_FILE = DATA_DIR / "customers_synthetic.csv"
OUTPUT_FILE = DATA_DIR / "customers_features.parquet"

# 2. Create features from raw customer data
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "customer_id",
        "customer_type",
        "monthly_income",
        "app_sessions_30d",
        "product_views_30d",
        "days_since_last_login",]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    features = df.copy()
    # Feature 1: Engagement score
    features["engagement_score"] = (
        np.log1p(features["app_sessions_30d"])
        + 0.7 * np.log1p(features["product_views_30d"])
        - 0.04 * features["days_since_last_login"])
    # Feature 2: Income band
    income_bins = [-np.inf,25000,50000,100000,200000,np.inf,]
    income_labels = ["low","lower_middle","middle","upper_middle","high",]
    features["income_band"] = pd.cut(features["monthly_income"],bins=income_bins,labels=income_labels,right=False,).astype("string")
    return features

# 3. Run the pipeline
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Reading raw customer data...")
    raw_data = pd.read_csv(INPUT_FILE)
    print(f"Raw dataset shape: {raw_data.shape}")
    print("Building features...")
    feature_data = build_features(raw_data)
    print(f"Processed dataset shape: {feature_data.shape}")
    print("Saving processed data as Parquet...")
    feature_data.to_parquet(OUTPUT_FILE,index=False,engine="pyarrow",)

    # Verify that Parquet can be read back
    reloaded_data = pd.read_parquet(OUTPUT_FILE, engine="pyarrow",)
    assert len(reloaded_data) == len(raw_data)
    assert reloaded_data["customer_id"].nunique() == len(raw_data)
    print("Feature engineering completed successfully.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(reloaded_data)}")
    print(f"Total columns: {len(reloaded_data.columns)}")

if __name__ == "__main__":
    main()
