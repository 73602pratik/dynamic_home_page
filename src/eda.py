from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customers_synthetic.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    # Summary of numeric features
    numeric_summary = df.describe(include="all").transpose()
    numeric_summary.to_csv(REPORTS_DIR / "descriptive_summary.csv")

    # Target prevalence: useful before choosing model metrics
    targets = ["applied_pl_90d", "applied_sme_90d", "applied_b2b_90d"]
    target_prevalence = {
        target: {
            "positive_count": int(df[target].sum()),
            "positive_rate": round(float(df[target].mean()), 4),
            "negative_count": int((df[target] == 0).sum()),
        }
        for target in targets
    }
    (REPORTS_DIR / "target_prevalence.json").write_text(
        json.dumps(target_prevalence, indent=2), encoding="utf-8"
    )

    # Customer type distribution
    type_counts = df["customer_type"].value_counts().sort_index()
    type_counts.to_csv(REPORTS_DIR / "customer_type_distribution.csv", header=["count"])

    # Simple chart of simulated target prevalence
    rates = [target_prevalence[target]["positive_rate"] * 100 for target in targets]
    labels = ["Personal Loan", "SME", "B2B"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, rates)
    ax.set_title("Synthetic application target prevalence")
    ax.set_ylabel("Positive labels (%)")
    ax.set_ylim(0, max(rates) * 1.25 if max(rates) > 0 else 1)
    ax.bar_label(bars, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "target_prevalence.png", dpi=150)
    plt.close(fig)

    print("EDA complete. Files written to reports/")
    print("\nCustomer type counts:")
    print(type_counts.to_string())
    print("\nSynthetic target prevalence:")
    for target, stats in target_prevalence.items():
        print(f"- {target}: {stats['positive_rate']:.1%} positive ({stats['positive_count']} rows)")
    print("\nReminder: these are simulated labels, not real business conversion rates.")

if __name__ == "__main__":
    main()
