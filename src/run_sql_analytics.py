
from pathlib import Path

import duckdb
import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "reports" / "end_to_end_homepage_results.csv"
SQL_FILE = ROOT / "sql" / "customer_analytics.sql"
OUTPUT_DIR = ROOT / "reports" / "sql_analytics"


# =========================================================
# 2. CHECK INPUT FILES
# =========================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "End-to-end homepage results not found. "
        "Run test_end_to_end.py first."
    )

if not SQL_FILE.exists():
    raise FileNotFoundError(
        f"SQL file not found: {SQL_FILE}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3. LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("          CUSTOMER PERSONALIZATION SQL ANALYTICS")
print("=" * 65)

print(f"\nCustomers loaded: {len(df):,}")

connection = duckdb.connect(database=":memory:")
connection.register("homepage_results", df)


# =========================================================
# 4. READ AND EXECUTE SQL QUERIES
# =========================================================

sql_text = SQL_FILE.read_text(encoding="utf-8")

queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

for index, query in enumerate(queries, start=1):

    # Skip comment-only statements.
    executable_lines = [
        line for line in query.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]

    if not executable_lines:
        continue

    result = connection.execute(query).df()

    print("\n" + "=" * 65)
    print(f"SQL ANALYSIS {index}")
    print("=" * 65)

    print(result.to_string(index=False))

    output_file = OUTPUT_DIR / f"analysis_{index}.csv"
    result.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file.name}")


# =========================================================
# 5. CLOSE CONNECTION
# =========================================================

connection.close()

print("\n" + "=" * 65)
print("SQL ANALYTICS COMPLETED SUCCESSFULLY")
print(f"Reports saved in: {OUTPUT_DIR}")
print("=" * 65)
