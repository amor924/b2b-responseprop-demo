from pathlib import Path
import joblib
import json
import pandas as pd

from .data import make_train_test
from .features import build_pipeline


def main():
    repo_root = Path(__file__).resolve().parents[1]

    X_train, X_test, y_train, y_test = make_train_test()

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, models_dir / "model.joblib")

    metadata = {
        "model_name": "b2b_propensity_logistic",
        "model_type": "LogisticRegression",
        "train_rows": int(X_train.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "train_date": str(pd.Timestamp.utcnow()),
    }

    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
