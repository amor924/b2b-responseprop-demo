from pathlib import Path
import pandas as pd
import joblib

from .data import load_raw


def main():
    repo_root = Path(__file__).resolve().parents[1]

    pipe = joblib.load(repo_root / "models" / "model.joblib")

    # IMPORTANT: Use same feature engineering as training
    df = load_raw()

    X = df.drop(columns=["response_flag"], errors="ignore")

    df["predicted_probability"] = pipe.predict_proba(X)[:, 1]

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(reports_dir / "scored_output.csv", index=False)

    print("Scoring complete.")


if __name__ == "__main__":
    main()
