from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "customers_features.parquet"

TRAIN_FILE = DATA_DIR / "train.parquet"
VALIDATION_FILE = DATA_DIR / "validation.parquet"
TEST_FILE = DATA_DIR / "test.parquet"

RANDOM_STATE = 42
TARGET_FOR_STRATIFICATION = "applied_b2b_90d"


def split_dataset(df):
    """
    Split data into:
    70% training
    15% validation
    15% testing
    """

    if TARGET_FOR_STRATIFICATION not in df.columns:
        raise ValueError(
            f"Missing required column: {TARGET_FOR_STRATIFICATION}"
        )

    # First split: 70% training and 30% temporary data
    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_FOR_STRATIFICATION]
    )

    # Second split: divide temporary data equally
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_df[TARGET_FOR_STRATIFICATION]
    )

    return train_df, validation_df, test_df


def print_split_summary(name, df):
    print(f"\n{name} dataset")
    print("-" * 35)
    print(f"Rows: {len(df)}")
    print(
        "B2B application rate:",
        round(df[TARGET_FOR_STRATIFICATION].mean() * 100, 2),
        "%"
    )


def main():
    print("Reading feature dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run build_features.py first."
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Total records: {len(df)}")

    train_df, validation_df, test_df = split_dataset(df)

    # Save the three datasets
    train_df.to_parquet(TRAIN_FILE, index=False)
    validation_df.to_parquet(VALIDATION_FILE, index=False)
    test_df.to_parquet(TEST_FILE, index=False)

    print_split_summary("Training", train_df)
    print_split_summary("Validation", validation_df)
    print_split_summary("Testing", test_df)

    print("\nDataset splitting completed.")
    print(f"Saved training data: {TRAIN_FILE}")
    print(f"Saved validation data: {VALIDATION_FILE}")
    print(f"Saved testing data: {TEST_FILE}")


if __name__ == "__main__":
    main()