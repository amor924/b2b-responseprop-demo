from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from .data import make_train_test


def calculate_psi(expected, actual, bins=10):
    expected_perc, bins = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)

    psi = np.sum(
        (expected_perc - actual_perc) *
        np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
    )
    return psi


def main():
    repo_root = Path(__file__).resolve().parents[1]

    pipe = joblib.load(repo_root / "models" / "model.joblib")
    X_train, X_test, y_train, y_test = make_train_test()

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    psi_score = calculate_psi(p_train, p_test)

    print(f"PSI Score: {psi_score:.4f}")

    if psi_score > 0.2:
        print("⚠️ Significant drift detected.")


if __name__ == "__main__":
    main()
