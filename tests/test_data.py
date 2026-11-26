import unittest
from pathlib import Path
import pandas as pd
from src.validate_data import DATA_PATH, validate_data

class TestSyntheticCustomerData(unittest.TestCase):
    def test_dataset_passes_validation(self):
        self.assertEqual(validate_data(DATA_PATH), [])

    def test_target_columns_are_binary(self):
        df = pd.read_csv(DATA_PATH)
        for col in ["applied_pl_90d", "applied_sme_90d", "applied_b2b_90d"]:
            self.assertTrue(set(df[col].unique()).issubset({0, 1}), col)

    def test_customer_id_is_unique(self):
        df = pd.read_csv(DATA_PATH)
        self.assertEqual(df["customer_id"].nunique(), len(df))

if __name__ == "__main__":
    unittest.main()
