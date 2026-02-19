from pathlib import Path
import pandas as pd

NUMERIC_COLS = [
    "company_size",
    "revenue",
    "tenure_months",
    "contract_remaining_months",
    "current_product_count",
    "avg_monthly_bill",
    "email_open_rate",
    "website_visits_30d",
    "prior_campaign_engagement",
]

def main():
    repo_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(repo_root / "data" / "raw" / "b2b_marketing_raw_dataset.csv")

    y = df["response_flag"].astype(int)

    out_dir = repo_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tables = []

    for col in NUMERIC_COLS:
        # quantile bins (10 by default); duplicates='drop' handles low-variance columns
        b = pd.qcut(df[col], q=10, duplicates="drop")
        t = (
            pd.DataFrame({"bin": b, "y": y})
            .groupby("bin", observed=True)
            .agg(n=("y", "size"), response_rate=("y", "mean"))
            .reset_index()
        )
        t["feature"] = col
        all_tables.append(t)

    out = pd.concat(all_tables, ignore_index=True)
    out.to_csv(out_dir / "eda_binning_tables.csv", index=False)

    print(f"Saved: {out_dir / 'eda_binning_tables.csv'}")

if __name__ == "__main__":
    main()
