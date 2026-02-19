from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_raw():
    repo_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(repo_root / "data" / "raw" / "b2b_marketing_raw_dataset.csv")

    # --- Engineered features ---
    df["renewal_window_flag"] = (df["contract_remaining_months"] <= 3).astype(int)
    df["high_engagement_flag"] = (df["website_visits_30d"] >= 5).astype(int)
    df["large_account_flag"] = (df["company_size"] >= 800).astype(int)
    df["revenue_per_employee"] = df["revenue"] / df["company_size"]

    # --- EDA-driven binning ---
    df["company_size_bin"] = pd.cut(
        df["company_size"],
        bins=[0, 200, 800, df["company_size"].max()],
        labels=["small", "mid", "large"],
    )

    df["website_visits_bin"] = pd.cut(
        df["website_visits_30d"],
        bins=[-1, 3, df["website_visits_30d"].max()],
        labels=["low", "high"],
    )

    df["contract_remaining_bin"] = pd.cut(
        df["contract_remaining_months"],
        bins=[-1, 12, 24, df["contract_remaining_months"].max()],
        labels=["short", "medium", "long"],
    )

    return df


def make_train_test():
    df = load_raw()

    X = df.drop(columns=["response_flag"])
    y = df["response_flag"]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
